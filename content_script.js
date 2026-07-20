// モデルは44.1kHz学習なので、コンテキスト生成時にサンプルレートを指定する
// (sampleRateプロパティへの代入は読み取り専用のため無効)
let audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });

let stopped = false;
let source = null;
let lastVideo = null;
let workletNode;
let videoMirror = null;
let videoSyncEnabled = true; // 映像を音声処理の遅延分だけ遅らせてリップシンクを合わせる

chrome.storage.local.get({ videoSync: true }, (v) => {
    videoSyncEnabled = v.videoSync;
});
chrome.storage.onChanged.addListener((changes) => {
    if (changes.videoSync) {
        videoSyncEnabled = changes.videoSync.newValue;
        if (!videoSyncEnabled && videoMirror) {
            videoMirror.destroy();
            videoMirror = null;
        }
        if (videoSyncEnabled && !stopped && lastVideo && !videoMirror) {
            videoMirror = new VideoDelayMirror(lastVideo);
        }
    }
});

/**
 * 映像遅延ミラー
 *
 * 音声はAI処理のパイプライン遅延 (数百ms) が乗るため、そのままだと映像が先行する。
 * <video>の映像だけを遅らせることはできないので、フレームをキャプチャして
 * video の直上に重ねた canvas へ遅延描画することでリップシンクを合わせる。
 * 遅延量は worklet が実測した値に自動追従する。
 */
class VideoDelayMirror {
    constructor(video) {
        this.video = video;
        this.delayMs = 0;
        this.targetDelayMs = 0;
        this.queue = []; // { bmp, t }
        this.running = true;

        this.canvas = document.createElement("canvas");
        this.canvas.style.position = "absolute";
        this.canvas.style.pointerEvents = "none"; // 操作はvideo/コントロールに透過
        this.canvas.style.display = "none";
        this.ctx = this.canvas.getContext("2d");
        if (video.parentElement) {
            // videoの直後の兄弟にすると、動画サイトのコントロール類 (通常より高いz) の下に入る
            video.parentElement.insertBefore(this.canvas, video.nextSibling);
        }

        this.captureSupported = typeof video.requestVideoFrameCallback === "function";
        this.scheduleCapture();
        this.renderLoop();
    }

    setDelay(ms) {
        this.targetDelayMs = ms;
    }

    scheduleCapture() {
        if (!this.running) return;
        const capture = async () => {
            if (!this.running) return;
            try {
                const bmp = await createImageBitmap(this.video);
                this.queue.push({ bmp, t: performance.now() });
                // 溜まりすぎ防止 (約2秒 @60fps)
                while (this.queue.length > 120) this.queue.shift().bmp.close();
            } catch (e) {
                // DRM保護された動画などはキャプチャ不可。ミラーは諦めて非表示のまま
            }
            this.scheduleCapture();
        };
        if (this.captureSupported) {
            this.video.requestVideoFrameCallback(capture);
        } else {
            requestAnimationFrame(capture);
        }
    }

    renderLoop() {
        if (!this.running) return;
        requestAnimationFrame(() => this.renderLoop());

        // 遅延量の急変はカクつくので徐々に追従させる
        this.delayMs += (this.targetDelayMs - this.delayMs) * 0.05;

        if (this.delayMs < 40) {
            // ほぼ遅延なし (停止中・起動直後など) はミラー不要
            this.canvas.style.display = "none";
            return;
        }

        // 位置とサイズをvideoに追従させる
        const v = this.video;
        if (v.parentElement && this.canvas.parentElement != v.parentElement) {
            v.parentElement.insertBefore(this.canvas, v.nextSibling);
        }
        this.canvas.style.left = v.offsetLeft + "px";
        this.canvas.style.top = v.offsetTop + "px";
        this.canvas.style.width = v.offsetWidth + "px";
        this.canvas.style.height = v.offsetHeight + "px";

        // 遅延時刻に最も近い過去フレームを選び、それより古いものは破棄
        const target = performance.now() - this.delayMs;
        let frame = null;
        while (this.queue.length > 0 && this.queue[0].t <= target) {
            if (frame) frame.bmp.close();
            frame = this.queue.shift();
        }
        if (frame) {
            if (this.canvas.width != frame.bmp.width || this.canvas.height != frame.bmp.height) {
                this.canvas.width = frame.bmp.width;
                this.canvas.height = frame.bmp.height;
            }
            this.ctx.drawImage(frame.bmp, 0, 0);
            frame.bmp.close();
            this.canvas.style.display = "";
        }
    }

    destroy() {
        this.running = false;
        for (const f of this.queue) f.bmp.close();
        this.queue = [];
        this.canvas.remove();
    }
}

chrome.runtime.onMessage.addListener((mes, _sender, sendResponse) => {
    if (mes.type == "stop") {
        stopped = true;
        if (workletNode) workletNode.port.postMessage({ type: "set_stop", value: true });
        if (videoMirror) {
            videoMirror.destroy();
            videoMirror = null;
        }
        sendResponse();
    } else if (mes.type == "start") {
        let video = document.querySelector("video");
        if (video == null) {
            // 動画がないページでは開始しない (popup側でエラー表示される)
            sendResponse("novideo");
            return;
        }
        stopped = false;
        startHookVideo(video);

        if (workletNode) workletNode.port.postMessage({ type: "set_stop", value: false });
        if (videoSyncEnabled && !videoMirror) {
            videoMirror = new VideoDelayMirror(video);
        }
        sendResponse("ok");
    } else if (mes.type == "buffer") {
        // 処理済み音声をworkletへ転送 (Transferableでコピーを回避)
        if (workletNode) {
            const l = new Float32Array(mes.payload[0]);
            const r = new Float32Array(mes.payload[1]);
            workletNode.port.postMessage({
                type: "processed_buffer",
                payload: [l, r],
                timelineEnd: mes.payload[2]
            }, [l.buffer, r.buffer]);
        }
        sendResponse();
    }
});

async function startHookVideo(target) {
    await audioContext.resume();
    await audioContext.audioWorklet.addModule(chrome.runtime.getURL("vocal-remover-processor.js"));

    if (source == null && target != lastVideo) {
        source = audioContext.createMediaElementSource(target);
        lastVideo = target;
        workletNode = new AudioWorkletNode(audioContext, "vocal-remover-processor");
        workletNode.port.onmessage = (event) => {
            if (event.data.type === "latency") {
                // workletが実測した音声遅延に映像ミラーを追従させる
                if (videoMirror) videoMirror.setDelay(event.data.value * 1000);
            } else if (event.data.type === "input_data") {
                // workletから届いたバッチをoffscreen(sandbox)へ直接プッシュする
                // (service workerによるポーリングは廃止。拡張機能メッセージは
                //  JSONシリアライズのため通常配列に変換して送る)
                chrome.runtime.sendMessage({
                    type: "inputBuffer",
                    payload: [Array.from(event.data.payload[0]), Array.from(event.data.payload[1])]
                });
            }
        };
        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
    }
}
