const toggleButton = document.querySelector("#toggle");
const statusText = document.querySelector("#status-text");
const messageArea = document.querySelector("#message");
const defaultMessage = messageArea.textContent;

let running = false;

function renderState() {
    document.body.classList.toggle("running", running);
    statusText.textContent = running ? "動作中" : "停止中";
    toggleButton.textContent = running ? "■  停止" : "▶  実行";
}

function showError(text) {
    messageArea.textContent = text;
    messageArea.classList.add("error");
    setTimeout(() => {
        messageArea.textContent = defaultMessage;
        messageArea.classList.remove("error");
    }, 3000);
}

// 現在の動作状態をservice workerから取得して表示に反映
// (service worker起動直後などで応答が取れなかった場合は一度だけ再試行)
function fetchStatus(attempt) {
    chrome.runtime.sendMessage({ type: "getStatus" }, (res) => {
        if (chrome.runtime.lastError || res == null) {
            if (attempt < 1) setTimeout(() => fetchStatus(attempt + 1), 200);
            return;
        }
        running = !!res.running;
        renderState();
    });
}
fetchStatus(0);

function startOnTab(tabId, retried) {
    chrome.tabs.sendMessage(tabId, { type: "start", payload: null }, (responseFromContent) => {
        if (chrome.runtime.lastError) {
            // content scriptが未注入 (拡張機能更新直後の開きっぱなしのタブなど)。
            // その場で注入してから一度だけ再試行する。
            if (!retried) {
                chrome.scripting.executeScript({ target: { tabId: tabId }, files: ["content_script.js"] }, () => {
                    if (chrome.runtime.lastError) {
                        showError("このページでは実行できません。");
                        return;
                    }
                    startOnTab(tabId, true);
                });
            } else {
                showError("このページでは実行できません。");
            }
            return;
        }
        if (responseFromContent != "ok") {
            showError("再生中の動画が見つかりませんでした。動画のあるページでお試しください。");
            return;
        }
        chrome.runtime.sendMessage({ type: "tabId", payload: tabId });
        running = true;
        renderState();
    });
}

toggleButton.addEventListener("click", () => {
    if (running) {
        chrome.runtime.sendMessage({ type: "stop", payload: null });
        running = false;
        renderState();
    } else {
        // 対象のタブを特定して、コンテンツに開始を通知
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            startOnTab(tabs[0].id, false);
        });
    }
});


// ミックス割合スライダー
// (ボーカル0% + 伴奏100% = 従来動作、ボーカル100% + 伴奏0% = ボーカルのみ)
const vocalSlider = document.querySelector("#vocal-ratio");
const vocalValue = document.querySelector("#vocal-ratio-value");
const instSlider = document.querySelector("#inst-ratio");
const instValue = document.querySelector("#inst-ratio-value");

// トラックの塗りつぶし位置をスライダー値に同期させる
function updateFill(slider) {
    slider.style.setProperty("--p", slider.value + "%");
}

// 前回の設定値を復元
chrome.storage.local.get({ vocalRatio: 0, instRatio: 100 }, (v) => {
    vocalSlider.value = v.vocalRatio;
    vocalValue.textContent = v.vocalRatio;
    instSlider.value = v.instRatio;
    instValue.textContent = v.instRatio;
    updateFill(vocalSlider);
    updateFill(instSlider);
});

function onRatioChange() {
    const vocal = Number(vocalSlider.value);
    const inst = Number(instSlider.value);
    vocalValue.textContent = vocal;
    instValue.textContent = inst;
    updateFill(vocalSlider);
    updateFill(instSlider);
    chrome.storage.local.set({ vocalRatio: vocal, instRatio: inst });
    // offscreenへ即時反映 (offscreenが現在値を保持してsandboxへ伝える)
    chrome.runtime.sendMessage({ type: "mixRatio", payload: { vocal: vocal, inst: inst } });
}

vocalSlider.addEventListener("input", onRatioChange);
instSlider.addEventListener("input", onRatioChange);


// ノイズ除去トグル (MDXモデルのノイズフロア対策。推論が2倍になるためGPU負荷増)
// FirefoxはWebGPU性能がChromeの約3倍遅くdenoise ONだとリアルタイム処理が
// 厳しいため、デフォルトOFFにする (browserグローバルの有無でFirefoxを判定)
const DENOISE_DEFAULT = typeof browser === "undefined"; // Chrome: true / Firefox: false
const denoiseToggle = document.querySelector("#denoise");

chrome.storage.local.get({ denoise: DENOISE_DEFAULT }, (v) => {
    denoiseToggle.checked = v.denoise;
});

denoiseToggle.addEventListener("change", () => {
    chrome.storage.local.set({ denoise: denoiseToggle.checked });
    chrome.runtime.sendMessage({ type: "denoise", payload: denoiseToggle.checked });
});
