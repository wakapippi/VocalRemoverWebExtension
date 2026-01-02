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

/**
 * 推論メイン処理
 */
async function runModel() {
    tf.engine().startScope();

    // AI処理に必要なサンプル数(31744)に達するまで待機
    if (buffer0.length < 31744) {
        await waitMs(10);
        tf.engine().endScope();
        return;
    }

    let currentLast = totalBufferPointer;

    // 最新のセグメントを切り出し
    let originalLeft = buffer0.slice(buffer0.length - 31744, buffer0.length);
    let originalRight = buffer1.slice(buffer1.length - 31744, buffer1.length);

    // 前後のパディング (境界ノイズ対策)
    let padSize = 3072;
    let left = new Array(padSize).fill(0).concat(originalLeft).concat(new Array(padSize).fill(0));
    let right = new Array(padSize).fill(0).concat(originalRight).concat(new Array(padSize).fill(0));

    // 入力バッファの整理
    buffer0.splice(0, buffer0.length - 31744 - 1);
    buffer1.splice(0, buffer1.length - 31744 - 1);

    let inputL = tf.tensor1d(left);
    let inputR = tf.tensor1d(right);

    // AIに渡す低域(l, r)と、保存用の高域(highFreqL, R)を取得
    const { low: l, high: highFreqL } = getStftParts(inputL);
    const { low: r, high: highFreqR } = getStftParts(inputR);

    // AIモデルへの入力成形 [1, 4, 3072, 32]
    let t0 = tf.stack([l, r], 3);
    let t1 = t0.transpose([0, 3, 1, 2]);
    let reshaped2 = t1.reshape([1, 4, -1, t1.shape[t1.shape.length - 1]]);

    if (!window.model) {
        tf.engine().endScope();
        return;
    }

    // AI推論実行
    let prediction = model.predict(reshaped2);

    // --- 後処理: AIの出力と元音の高域を結合してISTFT ---
    let signals = await istft(prediction, highFreqL, highFreqR);

    // 出力バッファへ書き込み
    let elapse = currentLast - writtenPointer;
    let outdata0 = Array.from(signals[0]).slice(padSize, padSize + 31744);
    let outdata1 = Array.from(signals[1]).slice(padSize, padSize + 31744);

    if (elapse < outdata0.length) {
        writeBuffer0 = writeBuffer0.concat(outdata0.slice(outdata0.length - elapse));
        writeBuffer1 = writeBuffer1.concat(outdata1.slice(outdata1.length - elapse));
    } else {
        writeBuffer0 = writeBuffer0.concat(outdata0);
        writeBuffer1 = writeBuffer1.concat(outdata1);
    }

    writtenPointer = currentLast;
    tf.engine().endScope();
}

/**
 * STFTを低域と高域に分けて取得
 */
function getStftParts(input) {
    return tf.tidy(() => {
        let res = tf.signal.stft(input, 6144, 1024, 6144, createHannWindow);
        let real = tf.real(res);
        let imag = tf.imag(res);

        // 低域 (AI用: 0〜3071番目のビン)
        let lowReal = tf.slice(real, [0, 0], [32, 3072]).transpose();
        let lowImag = tf.slice(imag, [0, 0], [32, 3072]).transpose();
        let low = tf.stack([lowReal, lowImag], 0);

        // 高域 (保存用: 3072番目以降のビン)
        let highReal = tf.slice(real, [0, 3072], [32, -1]).transpose();
        let highImag = tf.slice(imag, [0, 3072], [32, -1]).transpose();
        let high = tf.stack([highReal, highImag], 0);

        return { low, high };
    });
}

/**
 * 逆短時間フーリエ変換 (高域結合 + ISTFT)
 */
async function istft(x, highFreqL, highFreqR) {
    const n = 6144 / 2 + 1; // 3073 bins

    // AI出力を [Channel, Real/Imag, Freq, Time] に戻す
    let x_reshaped = x.reshape([2, 2, 3072, 32]);

    let combined = tf.tidy(() => {
        // AIのL/Rと、元音のL/R高域をそれぞれ結合
        let l_full = tf.concat([x_reshaped.slice([0, 0, 0, 0], [1, 2, 3072, 32]).squeeze([0]), highFreqL], 1);
        let r_full = tf.concat([x_reshaped.slice([1, 0, 0, 0], [1, 2, 3072, 32]).squeeze([0]), highFreqR], 1);
        return tf.stack([l_full, r_full], 0); // [2, 2, 3073, 32]
    });

    let x3 = combined.reshape([-1, 2, n, 32]);
    let x4 = x3.transpose([0, 2, 3, 1]); // [Batch*Channel, Freq, Time, Real/Imag]
    
    const splitTensors = tf.split(x4, 2, 0);
    let l_spec = tf.squeeze(splitTensors[0], [0]);
    let r_spec = tf.squeeze(splitTensors[1], [0]);

    // 正規化付きのISTFT実行
    let result = await getIstftLR_Normalized(l_spec, r_spec, 6144, 1024, createHannWindow);
    return result;
}


async function getIstftLR_Normalized(spectrogramL, spectrogramR, fftSize, hopSize, windowFn) {
    const numFrames = spectrogramL.shape[1];
    const size = fftSize + hopSize * (numFrames - 1);
    
    let outputArrayL = new Float32Array(size);
    let outputArrayR = new Float32Array(size);
    let weightArray = new Float32Array(size); // 窓の重み合計用

    const windowTensor = windowFn(fftSize);
    const windowValues = await windowTensor.data();

    let tasks = [];
    let needsToDispose = [];

    for (let g = 0; g < 2; g++) {
        let spectrogram = (g == 0) ? spectrogramL : spectrogramR;

        for (let i = 0; i < numFrames; i++) {
            let frame = spectrogram.slice([0, i], [-1, 1]);
            let frame2 = tf.squeeze(frame);
            let realPart = frame2.slice([0, 0], [-1, 1]);
            let imagPart = frame2.slice([0, 1], [-1, 1]);
            let complexTensor = tf.complex(realPart, imagPart);
            
            let timeDomainFrame = tf.spectral.irfft(tf.squeeze(complexTensor));
            let windowedFrame = tf.mul(timeDomainFrame, windowTensor);

            needsToDispose.push(windowedFrame);
            tasks.push(windowedFrame.data());
            
            // テンソルの解放
            frame.dispose(); frame2.dispose(); realPart.dispose();
            imagPart.dispose(); complexTensor.dispose(); timeDomainFrame.dispose();
        }
    }

    let completedTasks = await Promise.all(tasks);
    let framesL = completedTasks.slice(0, numFrames);
    let framesR = completedTasks.slice(numFrames);

    // 重ね合わせ加算と重み計算
    for (let i = 0; i < numFrames; i++) {
        const startIdx = i * hopSize;
        const dataL = framesL[i];
        const dataR = framesR[i];

        for (let j = 0; j < fftSize; j++) {
            if (startIdx + j < size) {
                outputArrayL[startIdx + j] += dataL[j];
                outputArrayR[startIdx + j] += dataR[j];
                // LとR共通の窓重みを蓄積 (STFTとISTFTで2回掛けているので2乗)
                if (i < numFrames) {
                    weightArray[startIdx + j] += windowValues[j] ** 2;
                }
            }
        }
    }

    // 正規化の実行 (窓の重なりによる音量変動をキャンセル)
    for (let i = 0; i < size; i++) {
        if (weightArray[i] > 1e-10) {
            outputArrayL[i] /= weightArray[i];
            outputArrayR[i] /= weightArray[i];
        }
    }

    // メモリ解放
    for (const t of needsToDispose) t.dispose();
    windowTensor.dispose();

    return [outputArrayL, outputArrayR];
}

function createHannWindow(length) {
    return tf.tidy(() => tf.signal.hannWindow(length));
}