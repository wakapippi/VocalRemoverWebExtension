const SEGMENT_SIZE = 31744;   // AI処理に必要なサンプル数 (32フレーム分)
const PAD_SIZE = 3072;        // 境界ノイズ対策の前後ゼロパディング
const INPUT_SIZE = SEGMENT_SIZE + PAD_SIZE * 2;
const MIN_NEW_SAMPLES = 3072; // 新規サンプルがこれ未満のうちは推論を見送る (約70ms)
const SILENCE_THRESHOLD = 1e-6; // これ以下の振幅しかなければ無音とみなす (約-120dB)

// 直近SEGMENT_SIZEサンプルだけを保持するスライディングウィンドウ
// (concat/spliceによる巨大配列の作り直しを避ける)
let window0 = new Float32Array(SEGMENT_SIZE);
let window1 = new Float32Array(SEGMENT_SIZE);
let filled = 0;         // ウィンドウ内の有効サンプル数
let totalReceived = 0;  // 累計受信サンプル数
let writtenPointer = 0; // 出力済みサンプル位置

let source;
let origin;
let vocalRatio = 0;   // ボーカルのミックス割合 (%)
let instRatio = 100;  // 伴奏のミックス割合 (%)

window.addEventListener("message", (event) => {

    if (event.data.type == "buffer") {
        appendToWindow(event.data.payload[0], event.data.payload[1]);
        if (event.data.mixRatio != null) {
            vocalRatio = event.data.mixRatio.vocal;
            instRatio = event.data.mixRatio.inst;
        }
        source = event.source;
        origin = event.origin;
    }
});

function appendToWindow(chunk0, chunk1) {
    const n = chunk0.length;
    if (n >= SEGMENT_SIZE) {
        window0.set(chunk0.subarray(n - SEGMENT_SIZE));
        window1.set(chunk1.subarray(n - SEGMENT_SIZE));
        filled = SEGMENT_SIZE;
    } else {
        const overflow = filled + n - SEGMENT_SIZE;
        if (overflow > 0) {
            window0.copyWithin(0, overflow, filled);
            window1.copyWithin(0, overflow, filled);
            filled -= overflow;
        }
        window0.set(chunk0, filled);
        window1.set(chunk1, filled);
        filled += n;
    }
    totalReceived += n;
}


(async function () {

    await waitMs(50);
    await tf.setBackend('webgpu');
    window.model = await tf.loadGraphModel("./models/model.json");

    while (true) {
        const result = await runModel();
        if (result && source) {
            source.postMessage(result, origin);
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
    // ウィンドウが埋まるまで、また新規サンプルが十分溜まるまで待機
    const currentLast = totalReceived;
    const elapse = currentLast - writtenPointer;
    if (filled < SEGMENT_SIZE || elapse < MIN_NEW_SAMPLES || !window.model) {
        await waitMs(10);
        return null;
    }

    const take = Math.min(elapse, SEGMENT_SIZE);

    // 無音区間 (一時停止中など) は推論をスキップしてGPU負荷を抑え、無音をそのまま返す
    if (isSilent(window0) && isSilent(window1)) {
        writtenPointer = currentLast;
        return [new Float32Array(take), new Float32Array(take)];
    }

    tf.engine().startScope();

    // 最新のセグメントに前後のゼロパディングを付けて切り出し (境界ノイズ対策)
    const left = new Float32Array(INPUT_SIZE);
    const right = new Float32Array(INPUT_SIZE);
    left.set(window0, PAD_SIZE);
    right.set(window1, PAD_SIZE);

    let inputL = tf.tensor1d(left);
    let inputR = tf.tensor1d(right);

    // AIに渡す低域(l, r)と、保存用の高域(highFreqL, R)を取得
    const { low: l, high: highFreqL } = getStftParts(inputL);
    const { low: r, high: highFreqR } = getStftParts(inputR);

    // AIモデルへの入力成形 [1, 4, 3072, 32]
    // UVRのSTFTと同じチャンネル優先の並び [L_re, L_im, R_re, R_im] に詰める
    let t0 = tf.stack([l, r], 0); // [ch, re/im, F, T]
    let reshaped2 = t0.reshape([1, 4, t0.shape[2], t0.shape[3]]);

    // AI推論実行
    let prediction = model.predict(reshaped2);

    // --- 後処理: AIの出力と元音の高域を結合してISTFT ---
    let signals = await istft(prediction, highFreqL, highFreqR);

    // 出力のうち、前回以降に到着した分だけを末尾から切り出す
    // (TypedArrayのsliceはコピーを返すのでpostMessageしても安全)
    const out0 = signals[0].slice(PAD_SIZE + SEGMENT_SIZE - take, PAD_SIZE + SEGMENT_SIZE);
    const out1 = signals[1].slice(PAD_SIZE + SEGMENT_SIZE - take, PAD_SIZE + SEGMENT_SIZE);

    // 伴奏(モデル出力)とボーカル(= 元音 - 伴奏)をそれぞれの割合でミックスする
    // 元音は推論開始時のスナップショット (left/right) から同じ位置を参照する
    const vR = vocalRatio / 100;
    const iR = instRatio / 100;
    if (vR != 0 || iR != 1) {
        const offset = PAD_SIZE + SEGMENT_SIZE - take;
        for (let i = 0; i < take; i++) {
            const inst0 = out0[i];
            const inst1 = out1[i];
            out0[i] = iR * inst0 + vR * (left[offset + i] - inst0);
            out1[i] = iR * inst1 + vR * (right[offset + i] - inst1);
        }
    }

    writtenPointer = currentLast;
    tf.engine().endScope();
    return [out0, out1];
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

/**
 * バッファ全体が無音かどうかの判定
 */
function isSilent(arr) {
    for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v > SILENCE_THRESHOLD || v < -SILENCE_THRESHOLD) return false;
    }
    return true;
}