// モデルは44.1kHz学習なので、コンテキスト生成時にサンプルレートを指定する
// (sampleRateプロパティへの代入は読み取り専用のため無効)
let audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });

let stopped = false;
let source = null;
let lastVideo = null;
let workletNode;

chrome.runtime.onMessage.addListener((mes, _sender, sendResponse) => {
    if (mes.type == "stop") {
        stopped = true;
        if (workletNode) workletNode.port.postMessage({ type: "set_stop", value: true });
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
        sendResponse("ok");
    } else if (mes.type == "buffer") {
        // 処理済み音声をworkletへ転送 (Transferableでコピーを回避)
        if (workletNode) {
            const l = new Float32Array(mes.payload[0]);
            const r = new Float32Array(mes.payload[1]);
            workletNode.port.postMessage({ type: "processed_buffer", payload: [l, r] }, [l.buffer, r.buffer]);
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
            if (event.data.type === "input_data") {
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
