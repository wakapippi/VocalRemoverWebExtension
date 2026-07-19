// offscreenドキュメントは既に存在する場合があるため、確認してから作成する
// (MV3のservice workerは再起動のたびにトップレベルが再実行される)
async function ensureOffscreenDocument() {
    const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
    if (contexts.length > 0) return;
    await chrome.offscreen.createDocument({
        url: chrome.runtime.getURL('offscreen.html'),
        reasons: ['IFRAME_SCRIPTING'],
        justification: 'Use WebGPU needs document API',
    });
}
ensureOffscreenDocument();

// service workerはアイドルで終了して変数が消えるため、対象タブはsession storageにも保持する
let currentTab = -1;
let restored = chrome.storage.session.get('currentTab').then((v) => {
    if (currentTab == -1 && v.currentTab != null) {
        currentTab = v.currentTab;
    }
});

// 入力音声(inputBuffer)はcontent scriptからoffscreenへ直接届くので、ここでは中継しない。
// service workerの役割は開始/停止の管理と、処理済み音声(outputBuffer)のタブへの中継のみ。
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type == "tabId") {
        if (currentTab != -1 && message.payload != currentTab) {
            chrome.tabs.sendMessage(currentTab, { type: "stop", payload: null }, () => chrome.runtime.lastError);
        }
        currentTab = message.payload;
        chrome.storage.session.set({ currentTab });
    } else if (message.type == "outputBuffer") {
        restored.then(() => {
            if (currentTab != -1) {
                chrome.tabs.sendMessage(currentTab, { type: "buffer", payload: message.payload }, () => chrome.runtime.lastError);
            }
        });
    } else if (message.type == "stop") {
        if (currentTab != -1) {
            chrome.tabs.sendMessage(currentTab, { type: "stop", payload: null }, () => chrome.runtime.lastError);
        }
        currentTab = -1;
        chrome.storage.session.remove('currentTab');
    } else if (message.type == "getMixRatio") {
        // offscreenはchrome.storageを使えないため、代わりに読み出して返す
        chrome.storage.local.get({ vocalRatio: 0, instRatio: 100 }, (v) => sendResponse({ vocal: v.vocalRatio, inst: v.instRatio }));
        return true; // 非同期でsendResponseを呼ぶ
    }
});
