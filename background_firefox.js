// Firefox用バックグラウンド (イベントページ)。
// Chrome版の service_worker.js (タブ管理・中継) と background.js (offscreen) を
// 統合したもの。FirefoxはOffscreen API非対応だがバックグラウンド頁にDOMが使える
// ため、sandbox.html を直接iframeでホストして中継ホップを1つ減らしている。

const sandboxFrame = document.getElementById("sandbox");

let currentTab = -1;
let vocalRatio = 0;
let instRatio = 100;
let denoiseEnabled = true;

// 設定の復元 (バックグラウンド頁はchrome.storageに直接アクセスできるため、
// Chrome版のようなservice worker経由の問い合わせは不要)
chrome.storage.local.get({ vocalRatio: 0, instRatio: 100, denoise: true }, (v) => {
    vocalRatio = v.vocalRatio;
    instRatio = v.instRatio;
    denoiseEnabled = v.denoise;
});

// イベントページはアイドルで終了して変数が消えるため、対象タブはsession storageにも保持する
let restored = chrome.storage.session.get('currentTab').then((v) => {
    if (currentTab == -1 && v.currentTab != null) {
        currentTab = v.currentTab;
    }
});

// sandboxからの処理済み音声 (Float32Array) をタブへ中継
window.addEventListener("message", (event) => {
    if (event.source != sandboxFrame.contentWindow) return;
    if (currentTab == -1) return;
    chrome.tabs.sendMessage(currentTab, {
        type: "buffer",
        payload: [Array.from(event.data[0]), Array.from(event.data[1])]
    }, () => chrome.runtime.lastError);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type == "inputBuffer") {
        // content scriptからの入力音声をsandboxへ渡す
        const l = new Float32Array(message.payload[0]);
        const r = new Float32Array(message.payload[1]);
        sandboxFrame.contentWindow.postMessage({
            type: "buffer",
            payload: [l, r],
            mixRatio: { vocal: vocalRatio, inst: instRatio },
            denoise: denoiseEnabled
        }, "*");
    } else if (message.type == "tabId") {
        if (currentTab != -1 && message.payload != currentTab) {
            chrome.tabs.sendMessage(currentTab, { type: "stop", payload: null }, () => chrome.runtime.lastError);
        }
        currentTab = message.payload;
        chrome.storage.session.set({ currentTab });
    } else if (message.type == "mixRatio") {
        vocalRatio = message.payload.vocal;
        instRatio = message.payload.inst;
    } else if (message.type == "denoise") {
        denoiseEnabled = message.payload;
    } else if (message.type == "stop") {
        if (currentTab != -1) {
            chrome.tabs.sendMessage(currentTab, { type: "stop", payload: null }, () => chrome.runtime.lastError);
        }
        currentTab = -1;
        chrome.storage.session.remove('currentTab');
        // メモリ解放のためsandboxをリロードする
        sandboxFrame.contentWindow.location.reload();
    } else if (message.type == "getStatus") {
        // popupが開いたときの状態表示用
        restored.then(() => {
            if (currentTab == -1) {
                sendResponse({ running: false });
                return;
            }
            chrome.tabs.get(currentTab, (tab) => {
                if (chrome.runtime.lastError || !tab) {
                    currentTab = -1;
                    chrome.storage.session.remove('currentTab');
                    sendResponse({ running: false });
                } else {
                    sendResponse({ running: true });
                }
            });
        });
        return true; // 非同期でsendResponseを呼ぶ
    }
});
