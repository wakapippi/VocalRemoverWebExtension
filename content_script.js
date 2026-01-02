let bufferL = [];
let bufferR = [];

let audioContext = new (window.AudioContext || window.webkitAudioContext)();
audioContext.sampleRate = 44100;

let bufferQueue = [];
let stopped = false;
let source = null;
let lastVideo = null;

chrome.runtime.onMessage.addListener((mes, _ev, sendResponse) => {
    if (mes.type == "stop") {
        stopped = true;
        if (workletNode) workletNode.port.postMessage({ type: "set_stop", value: true });
        sendResponse();
    }
    else if (mes.type == "start") {
        stopped = false;        
        let video = document.querySelector("video");
        startHookVideo(video);

        bufferQueue = [];
        bufferL = [];
        bufferR = [];
        if (workletNode) workletNode.port.postMessage({ type: "set_stop", value: false });
        sendResponse("ok");

    } else if (mes.type == "request") {
        if (bufferQueue.length == 0) {
            sendResponse();
        } else {
            sendResponse(bufferQueue.shift());
        }
    } else if (mes.type == "buffer") {
       if (workletNode) {
            workletNode.port.postMessage({
                type: "processed_buffer",
                payload: mes.payload
            });
        }
        sendResponse();
    }
});


let workletNode;

async function startHookVideo(target) {
    await audioContext.resume();
    await audioContext.audioWorklet.addModule(chrome.runtime.getURL("vocal-remover-processor.js"));

    if(source == null && target != lastVideo){
        source = audioContext.createMediaElementSource(target);
        lastVideo = target;
        workletNode = new AudioWorkletNode(audioContext, "vocal-remover-processor");
        workletNode.port.onmessage = (event) => {
            if (event.data.type === "input_data") {
                // 元のコードの bufferQueue.push に相当
                bufferQueue.push(event.data.payload);
            }
        };
        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
    }
}


/*

async function startHookVideo(target) {

    let source = audioContext.createMediaElementSource(target);

    let scriptNode = audioContext.createScriptProcessor(1024, source.channelCount, source.channelCount);
    scriptNode.addEventListener("audioprocess", processAudio);

    source.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

}


function processAudio(audioProcessingEvent) {
    // The input buffer is the song we loaded earlier
    let inputBuffer = audioProcessingEvent.inputBuffer;

    let l = Array.from(inputBuffer.getChannelData(0));
    let r;

    if (inputBuffer.numberOfChannels == 2) {
        r = Array.from(inputBuffer.getChannelData(1));
    } else {
        r = l;
    }

    let result = [l, r];

    bufferQueue.push(result);

    // The output buffer contains the samples that will be modified and played
    let outputBuffer = audioProcessingEvent.outputBuffer;

    // Loop through the output channels (in this case there is only one)
    for (let channel = 0; channel < outputBuffer.numberOfChannels; channel++) {

        let inputData = inputBuffer.getChannelData(channel);
        let outputData = outputBuffer.getChannelData(channel);

        if (stopped) {

            for (let sample = 0; sample < outputData.length; sample++) {

                outputData[sample] = inputData[sample];
            }

            continue;
        }

        let target = null;
        if (channel == 0) {
            target = bufferL;
        }
        if (channel == 1) {
            target = bufferR;
        }

        if (target != null && target.length >= outputData.length) {

            for (let sample = 0; sample < outputData.length; sample++) {

                outputData[sample] = target[sample];
            }

            target.splice(0, outputData.length);

        } else {
            console.warn("buffer underflow")
        }
    }
}
*/