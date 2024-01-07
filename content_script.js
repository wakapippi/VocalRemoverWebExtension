let bufferL = [];
let bufferR = [];

let audioContext = new (window.AudioContext || window.webkitAudioContext)();
audioContext.sampleRate = 44100;

let bufferQueue = [];
let stopped = false;

chrome.runtime.onMessage.addListener((mes, _ev, sendResponse) => {

    if (mes.type == "stop") {
        stopped = true;
        sendResponse();
    }
    else if (mes.type == "start") {

        stopped = false;

        let video = document.querySelector("video");
        startHookVideo(video);

        bufferQueue = [];
        bufferL = [];
        bufferR = [];

        sendResponse("ok");

    } else if (mes.type == "request") {
        if (bufferQueue.length == 0) {
            sendResponse();
        } else {
            sendResponse(bufferQueue.shift());
        }
    } else if (mes.type == "buffer") {

        bufferL = bufferL.concat(mes.payload[0]);
        bufferR = bufferR.concat(mes.payload[1]);
        sendResponse();
    }
    // ポップアップにメッセージを返す

});


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
