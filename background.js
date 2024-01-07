function process(data) {

    document.getElementById("sandbox").contentWindow.postMessage({ type: "buffer", payload: [data[0], data[1]] }, "*");
}

window.addEventListener("message", (event) => {
    let buffer = event.data;
    chrome.runtime.sendMessage({ type: "outputBuffer", payload: buffer });
});


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type == "inputBuffer") {
        process(message.payload);
    } else if (message.type == "stop") {
        // メモリ解放するためにoffscreenをreloadする。
        location.reload();
    }
});

async function waitMs(ms) {
    return new Promise((s, f) => {
        setTimeout(s, ms);
    });
}
