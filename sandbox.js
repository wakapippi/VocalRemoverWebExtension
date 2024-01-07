let buffer0 = [];
let buffer1 = [];

let totalBufferPointer = 0;

let writtenPointer = 0;

let writeBuffer0 = [];
let writeBuffer1 = [];
let source;
let origin;

window.addEventListener("message", async (event) => {

    if (event.data.type == "buffer") {
        let signal = event.data.payload;
        buffer0 = buffer0.concat(signal[0]);
        buffer1 = buffer1.concat(signal[1]);
        totalBufferPointer += signal[0].length;
        source = event.source;
        origin = event.origin;
    }
});


(async function () {

    await waitMs(50);
    await tf.setBackend('webgpu');
    window.model = await tf.loadGraphModel("./models/model.json");

    while (true) {
        await runModel();
        if (source) {
            source.postMessage([writeBuffer0, writeBuffer1], origin);
            writeBuffer0 = [];
            writeBuffer1 = [];
        }
    }
})();


async function waitMs(ms) {
    return new Promise((s, f) => {
        setTimeout(s, ms);
    });
}


async function runModel() {
    tf.engine().startScope()

    if (buffer0.length < 31744) {
        await waitMs(10);
        return;
    }

    let currentLast = totalBufferPointer;

    let originalLeft = buffer0.slice(buffer0.length - 31744, buffer0.length);
    let originalRight = buffer1.slice(buffer1.length - 31744, buffer1.length);


    let left = new Array(3072).fill(0).concat(originalLeft).concat(new Array(3072).fill(0));
    let right;
    if (buffer1.length >= 31744) {

        right = new Array(3072).fill(0).concat(originalRight).concat(new Array(3072).fill(0));
    } else {
        right = left.concat();
    }

    buffer0.splice(0, buffer0.length - 31744 - 1);
    buffer1.splice(0, buffer1.length - 31744 - 1);

    let inputL = tf.tensor1d(left);
    let inputR = tf.tensor1d(right);

    let l = getStft(inputL);
    let r = getStft(inputR);

    inputL.dispose();
    inputR.dispose();

    let t0 = tf.stack([l, r], 3);
    l.dispose();
    r.dispose();
    let t1 = t0.transpose([0, 3, 1, 2]);
    t0.dispose();

    let reshaped = t1.reshape([1, 2, 2, -1, t1.shape[t1.shape.length - 1]]);
    let reshaped2 = reshaped.reshape([1, 4, -1, t1.shape[t1.shape.length - 1]]);
    t1.dispose();
    reshaped.dispose();

    if (!window.model) {
        return;
    }

    let data = tf.tidy(() => {

        return model.predict(reshaped2);
    });

    reshaped2.dispose();

    let signals = await istft(data);
    let elapse = currentLast - writtenPointer;
    let outdata0 = Array.from(signals[0]);
    outdata0 = outdata0.splice(3072, 31744);
    let outdata1 = Array.from(signals[1]);
    outdata1 = outdata1.splice(3072, 31744);

    if (elapse < outdata0.length) {
        writeBuffer0 = writeBuffer0.concat(outdata0.slice(outdata0.length - elapse, outdata0.length));
    } else {
        writeBuffer0 = writeBuffer0.concat(outdata0);
    }

    if (elapse < outdata0.length) {
        writeBuffer1 = writeBuffer1.concat(outdata1.slice(outdata1.length - elapse, outdata1.length));
    } else {
        writeBuffer1 = writeBuffer1.concat(outdata1);
    }
    writtenPointer = currentLast;

    tf.dispose(data);


}


async function istft(x) {

    const n = 6144 / 2 + 1;
    const f_pad = tf.zeros([1, 4, 1, 32]);
    let x1 = tf.concat([x, f_pad], x.shape.length - 2);
    f_pad.dispose();
    let x2 = x1.reshape([1, 2, 2, 3073, 32])
    x1.dispose();
    let x3 = x2.reshape([-1, 2, n, 32]);
    x2.dispose();
    let x4 = x3.transpose([0, 2, 3, 1]);
    x3.dispose();
    const splitTensors = tf.split(x4, 2, 0);
    x4.dispose();
    let l = splitTensors[0];
    let l2 = tf.squeeze(l, [0]);
    let r = splitTensors[1];
    let r2 = tf.squeeze(r, [0]);
    let result = await getIstftLR(l2, r2, 6144, 1024, createHannWindow);

    l.dispose();
    l2.dispose();
    r.dispose();
    r2.dispose();
    tf.engine().endScope();
    return result;

}


function getStft(input) {

    return tf.tidy(() => {
        let res = tf.signal.stft(input, 6144, 1024, 6144, createHannWindow);
        let real = tf.real(res);
        let imag = tf.imag(res);
        real = tf.slice(real, [0, 0], [32, 3072]);
        imag = tf.slice(imag, [0, 0], [32, 3072]);
        real = real.transpose();
        imag = imag.transpose();
        let output = tf.stack([real, imag], 0);
        let zeroLike = tf.zerosLike(output);
        let isNaN = tf.isNaN(output);
        return tf.where(isNaN, zeroLike, output);
    });
}

async function getIstftLR(spectrogramL, spectrogramR, fftSize, hopSize, windowFn) {

    let tasks = [];
    let needsToDispose = [];
    const numFrames = spectrogramL.shape[1];
    let size = fftSize + hopSize * (numFrames - 1);
    let outputArrayL = new Float32Array(size);
    let outputArrayR = new Float32Array(size);

    for (let g = 0; g < 2; g++) {

        let spectrogram;
        if (g == 0) {
            spectrogram = spectrogramL;
        } else {
            spectrogram = spectrogramR;
        }

        const numFrames = spectrogram.shape[1];
        const numFrequencyBins = spectrogram.shape[0];
        let window = windowFn ? windowFn(fftSize) : tf.ones([fftSize]);

        for (let i = 0; i < numFrames; i++) {
            let frame = spectrogram.slice([0, i], [numFrequencyBins, 1]);
            let frame2 = tf.squeeze(frame);
            let realPart = frame2.slice([0, 0], [-1, 1]);
            let imagPart = frame2.slice([0, 1], [-1, 1]);
            let complexTensor = tf.complex(realPart, imagPart);
            let complexTensor2 = tf.squeeze(complexTensor)
            let timeDomainFrame = tf.spectral.irfft(complexTensor2);
            let timeDomainFrame2 = tf.mul(timeDomainFrame, window);

            frame.dispose();
            frame2.dispose();
            realPart.dispose();
            imagPart.dispose();
            complexTensor.dispose();
            complexTensor2.dispose();
            timeDomainFrame.dispose();
            needsToDispose.push(timeDomainFrame2);

            tasks.push(timeDomainFrame2.data());
        }

        window.dispose();
    }

    let completedTasks = await Promise.all(tasks);

    for (const iterator of needsToDispose) {
        iterator.dispose();
    }

    let framesL = completedTasks.splice(0, numFrames);
    let framesR = completedTasks;

    for (let i = 0; i < numFrames; i++) {
        const timeDomainArray = framesL[i];
        const startIdx = i * hopSize;
        for (let j = 0; j < fftSize; j++) {
            if (startIdx + j < outputArrayL.length) {
                outputArrayL[startIdx + j] += timeDomainArray[j];
            }
        }
    }
    for (let i = 0; i < numFrames; i++) {

        const timeDomainArray = framesR[i];
        const startIdx = i * hopSize;
        for (let j = 0; j < fftSize; j++) {
            if (startIdx + j < outputArrayR.length) {
                outputArrayR[startIdx + j] += timeDomainArray[j];
            }
        }
    }
    return [outputArrayL, outputArrayR];
}

function createHannWindow(length) {
    let ts = tf.tensor(tf.signal.hannWindow(length).dataSync());
    return ts;
}
