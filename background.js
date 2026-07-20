// ボーカル/伴奏のミックス割合 (%)。sandboxはstop時にリロードされるため、
// offscreen側で現在値を保持し、bufferメッセージに相乗りさせて常に伝える。
// offscreenドキュメントはchrome.storageに直接アクセスできないため、
// 保存値はservice worker経由で取得する。
let vocalRatio = 0;
let instRatio = 100;
let denoiseEnabled = true;
chrome.runtime.sendMessage({ type: "getMixRatio" }, (v) => {
    if (v) {
        vocalRatio = v.vocal;
        instRatio = v.inst;
        denoiseEnabled = v.denoise;
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type == "inputBuffer") {
        // content scriptからの入力音声をsandboxへ渡す
        // (iframeへのpostMessageは構造化クローンなのでFloat32Arrayのまま渡せる)
        const l = new Float32Array(message.payload[0]);
        const r = new Float32Array(message.payload[1]);
        document.getElementById("sandbox").contentWindow.postMessage({ type: "buffer", payload: [l, r], mixRatio: { vocal: vocalRatio, inst: instRatio }, denoise: denoiseEnabled }, "*");
    } else if (message.type == "mixRatio") {
        vocalRatio = message.payload.vocal;
        instRatio = message.payload.inst;
    } else if (message.type == "denoise") {
        denoiseEnabled = message.payload;
    } else if (message.type == "stop") {
        // メモリ解放するためにoffscreenをreloadする。
        location.reload();
    }
});

window.addEventListener("message", (event) => {
    // sandboxからの処理済み音声(Float32Array)を、拡張機能メッセージで送れる形に
    // 通常配列へ変換してservice worker経由でタブへ返す
    // (payload[2]は入力タイムライン位置。workletの遅延実測に使う)
    chrome.runtime.sendMessage({
        type: "outputBuffer",
        payload: [Array.from(event.data[0]), Array.from(event.data[1]), event.data[2]]
    });
});
