chrome.offscreen.createDocument({
    url: chrome.runtime.getURL('offscreen.html'),
    reasons: ['IFRAME_SCRIPTING'],
    justification: 'Use WebGPU needs document API',
});


let currentTab = -1;

async function startRequestLoop(id) {

    if (id == currentTab) {
        return;
    }

    currentTab = id;
    let promise = Promise.resolve();
    let suc;

    while (true) {
        promise = new Promise((s) => {
            suc = s;
        })
        chrome.tabs.sendMessage(id, { type: "request", payload: null }, (responseFromContent) => {
            suc(responseFromContent)
        });
        if (currentTab != id) {
            return;
        }
        let result = await promise;
        if (result != null) {
            process(result);
        }
    }

}

function process(data) {
    chrome.runtime.sendMessage({ type: "inputBuffer", payload: [data[0], data[1]] });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type == "tabId") {
        if (currentTab != -1 && message.payload != currentTab) {
            chrome.tabs.sendMessage(currentTab, { type: "stop", payload: message.payload }, (responseFromContent) => {

            });
        }
        startRequestLoop(message.payload);
    }
    else if (message.type == "outputBuffer") {
        chrome.tabs.sendMessage(currentTab, { type: "buffer", payload: message.payload }, (responseFromContent) => {

        });
    } else if (message.type == "stop") {
        chrome.tabs.sendMessage(currentTab, { type: "stop", payload: message.payload }, (responseFromContent) => {

        });
        currentTab = -1;
    }
});