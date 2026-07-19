// TF.js (WebGPU) と LiteRT.js で同一モデル・同一入力の推論時間を比較するベンチマーク。
// 実運用のパイプラインに合わせて、CPU上の入力データ → 推論 → CPUへの出力読み出し
// までの1往復を1回として計測する。
import { loadLiteRt, loadAndCompile, Tensor as LiteRtTensor } from "./node_modules/@litertjs/core/dist/index.js";

const SHAPE = [1, 4, 3072, 32];
const SIZE = SHAPE.reduce((a, b) => a * b, 1);
const WARMUP = 10;
const RUNS = 30;
const WASM_WARMUP = 3;
const WASM_RUNS = 10;

const logEl = document.getElementById("log");
const lines = [];
function log(s) {
    lines.push(s);
    logEl.textContent = lines.join("\n");
    console.log(s);
}

function stats(times) {
    const sorted = [...times].sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const pick = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
    return {
        mean: +mean.toFixed(2),
        median: +pick(0.5).toFixed(2),
        p10: +pick(0.1).toFixed(2),
        p90: +pick(0.9).toFixed(2),
        min: +sorted[0].toFixed(2),
    };
}

// 音声っぽいスケールの再現可能な疑似乱数入力
function makeInput(seed) {
    const data = new Float32Array(SIZE);
    let s = seed;
    for (let i = 0; i < SIZE; i++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        data[i] = ((s / 4294967296) - 0.5) * 0.2;
    }
    return data;
}

async function benchTfjs(model, input, warmup, runs, batch) {
    const shape = [batch, 4, 3072, 32];
    const data = batch == 1 ? input : (() => {
        const d = new Float32Array(SIZE * batch);
        for (let b = 0; b < batch; b++) d.set(input, b * SIZE);
        return d;
    })();

    const times = [];
    for (let i = 0; i < warmup + runs; i++) {
        const t0 = performance.now();
        const t = tf.tensor(data, shape);
        const out = model.predict(t);
        const result = await out.data(); // GPU完了までの同期を含む
        const t1 = performance.now();
        t.dispose();
        out.dispose();
        if (i >= warmup) times.push(t1 - t0);
        if (i == 0) window.__tfjsOut = result.slice(0, SIZE);
    }
    return stats(times);
}

async function benchLiteRt(model, input, warmup, runs) {
    const times = [];
    for (let i = 0; i < warmup + runs; i++) {
        const t0 = performance.now();
        const t = new LiteRtTensor(input, SHAPE);
        const outputs = await model.run(t);
        const out = outputs[0];
        const result = await out.data(); // GPU→CPUコピー (GPU完了までの同期を含む)
        const t1 = performance.now();
        t.delete();
        outputs.forEach((o) => o.delete());
        if (i >= warmup) times.push(t1 - t0);
        if (i == 0) window.__litertOut = result.slice(0, SIZE);
    }
    return stats(times);
}

function maxAbsDiff(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

(async function main() {
    const results = { ok: false };
    try {
        if (!navigator.gpu) throw new Error("WebGPU unavailable (navigator.gpu undefined)");
        const input = makeInput(12345);

        // --- TF.js WebGPU ---
        log("TF.js: setting webgpu backend...");
        await tf.setBackend("webgpu");
        await tf.ready();
        log("TF.js: loading graph model...");
        const tfjsModel = await tf.loadGraphModel("../models/model.json");

        log(`TF.js WebGPU: benchmarking batch=1 (${WARMUP} warmup + ${RUNS} runs)...`);
        results.tfjs_webgpu_b1 = await benchTfjs(tfjsModel, input, WARMUP, RUNS, 1);
        log("  -> " + JSON.stringify(results.tfjs_webgpu_b1));

        log(`TF.js WebGPU: benchmarking batch=2 (denoise相当)...`);
        results.tfjs_webgpu_b2 = await benchTfjs(tfjsModel, input, WARMUP, RUNS, 2);
        log("  -> " + JSON.stringify(results.tfjs_webgpu_b2));

        // --- LiteRT.js ---
        // Safari対応: GPUDevice.adapterInfo 未実装環境向けポリフィル
        // (LiteRTのEmscriptenバインディングが device.adapterInfo を参照して落ちる)
        if (typeof GPUDevice !== "undefined" && !("adapterInfo" in GPUDevice.prototype)) {
            log("polyfilling GPUDevice.adapterInfo (Safari)");
            const info = {
                vendor: "apple", architecture: "", device: "", description: "",
                subgroupMinSize: 32, subgroupMaxSize: 32,
            };
            Object.defineProperty(GPUDevice.prototype, "adapterInfo", { get: () => info });
        }
        log("LiteRT: loading wasm runtime...");
        await loadLiteRt("./node_modules/@litertjs/core/wasm/");

        log("LiteRT WebGPU: compiling model...");
        const liteGpu = await loadAndCompile("./vr_model.tflite", { accelerator: "webgpu" });
        log(`LiteRT WebGPU: benchmarking batch=1 (${WARMUP} warmup + ${RUNS} runs)...`);
        results.litert_webgpu_b1 = await benchLiteRt(liteGpu, input, WARMUP, RUNS);
        log("  -> " + JSON.stringify(results.litert_webgpu_b1));

        // 数値一致確認 (TF.js WebGPU vs LiteRT WebGPU)
        results.max_abs_diff = maxAbsDiff(window.__tfjsOut, window.__litertOut);
        log("output max abs diff (tfjs vs litert): " + results.max_abs_diff.toExponential(2));

        // --- LiteRT wasm (CPU/XNNPACK) 参考値 ---
        try {
            log("LiteRT Wasm(CPU): compiling model...");
            const liteCpu = await loadAndCompile("./vr_model.tflite", { accelerator: "wasm" });
            log(`LiteRT Wasm(CPU): benchmarking batch=1 (${WASM_WARMUP} warmup + ${WASM_RUNS} runs)...`);
            results.litert_wasm_b1 = await benchLiteRt(liteCpu, input, WASM_WARMUP, WASM_RUNS);
            log("  -> " + JSON.stringify(results.litert_wasm_b1));
        } catch (e) {
            log("LiteRT Wasm(CPU): skipped (" + e.message + ")");
        }

        results.ok = true;
        log("\n==== RESULT ====\n" + JSON.stringify(results, null, 2));
    } catch (e) {
        results.error = String(e && e.stack || e);
        log("ERROR: " + results.error);
    }
    window.__benchResult = results;
    window.__benchDone = true;
})();
