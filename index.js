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
