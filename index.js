document.querySelector("#execute").addEventListener("click", () => {

    // 対象のタブを特定して、コンテンツに送信
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { type: "start", payload: null }, (responseFromContent) => {
            // コンテンツから受け取ったレスポンスをダイアログ表示
            //alert(responseFromContent);
            // ポップアップを閉じる(コンテンツではない)
            //window.close();
            if (responseFromContent == "ok") {
                chrome.runtime.sendMessage({ type: "tabId", payload: tabs[0].id });
            }
        });
    });
});


document.querySelector("#stop").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "stop", payload: null });
});


// ミックス割合スライダー
// (ボーカル0% + 伴奏100% = 従来動作、ボーカル100% + 伴奏0% = ボーカルのみ)
const vocalSlider = document.querySelector("#vocal-ratio");
const vocalValue = document.querySelector("#vocal-ratio-value");
const instSlider = document.querySelector("#inst-ratio");
const instValue = document.querySelector("#inst-ratio-value");

// 前回の設定値を復元
chrome.storage.local.get({ vocalRatio: 0, instRatio: 100 }, (v) => {
    vocalSlider.value = v.vocalRatio;
    vocalValue.textContent = v.vocalRatio;
    instSlider.value = v.instRatio;
    instValue.textContent = v.instRatio;
});

function onRatioChange() {
    const vocal = Number(vocalSlider.value);
    const inst = Number(instSlider.value);
    vocalValue.textContent = vocal;
    instValue.textContent = inst;
    chrome.storage.local.set({ vocalRatio: vocal, instRatio: inst });
    // offscreenへ即時反映 (offscreenが現在値を保持してsandboxへ伝える)
    chrome.runtime.sendMessage({ type: "mixRatio", payload: { vocal: vocal, inst: inst } });
}

vocalSlider.addEventListener("input", onRatioChange);
instSlider.addEventListener("input", onRatioChange);
