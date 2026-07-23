// 4ステム分離 PoC
// kuielab_b per-stem MDXモデル4つで音声を Vocals/Drums/Bass/Other に分離する。
// ステムごとにn_fftが異なるが、hop=1024・32フレーム・セグメント31744サンプルを
// 共通にすることで4ステムの出力が時間軸で整列する。
import { loadLiteRt, loadAndCompile, getWebGpuDevice } from "@litertjs/core";
import { runWithTfjsTensors } from "./node_modules/@litertjs/tfjs-interop/dist/index.js";

const SR = 44100;
const HOP = 1024;
const DIM_F = 2048;
const FRAMES = 32;
const SEGMENT = HOP * (FRAMES - 1); // 31744
const XFADE = 1024;                 // セグメント間クロスフェード
const COMPENSATE = 1.035;           // kuielab系のcompensate値 (UVR model_data.json)

const STEMS = [
    { key: "vocals", label: "Vocals", nFft: 6144 },
    { key: "drums", label: "Drums", nFft: 4096 },
    { key: "bass", label: "Bass", nFft: 16384 },
    { key: "other", label: "Other", nFft: 8192 },
];

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const isAuto = new URLSearchParams(location.search).has("auto");

function setStatus(s) {
    statusEl.textContent = s;
    console.log(s);
}

// ============ 初期化 ============
let models = null;

async function init() {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    await loadLiteRt("./node_modules/@litertjs/core/wasm/");
    // LiteRTのGPUDeviceでTF.jsのwebgpuバックエンドを登録し直す (GPU上でテンソルを受け渡すため)
    const device = await getWebGpuDevice();
    tf.removeBackend("webgpu");
    tf.registerBackend("webgpu", () => new tf.WebGPUBackend(device, device.adapterInfo));
    await tf.setBackend("webgpu");

    models = {};
    for (const stem of STEMS) {
        setStatus(`モデル読み込み中... (${stem.key})`);
        models[stem.key] = await loadAndCompile(`./stems/kuielab_b_${stem.key}.tflite`, { accelerator: "webgpu" });
    }
    setStatus("準備完了。音声ファイルをドロップしてください。");
}

// ============ ステム1つ分のセグメント処理 ============
// ctxL/ctxR: SEGMENT + nFft サンプル (前後にnFft/2の実コンテキスト付き)
// 戻り値: [Float32Array, Float32Array] (SEGMENTサンプル)
async function processSegment(stem, ctxL, ctxR) {
    const nFft = stem.nFft;
    const pad = nFft / 2;
    const nBins = nFft / 2 + 1;

    tf.engine().startScope();
    try {
        const win = tf.signal.hannWindow(nFft);

        // STFT → 低域DIM_Fビンを [1,4,2048,32] (ch優先 [L_re,L_im,R_re,R_im]) に詰める
        const parts = [];
        for (const ctx of [ctxL, ctxR]) {
            const stft = tf.signal.stft(tf.tensor1d(ctx), nFft, HOP, nFft, () => win);
            const re = tf.slice(tf.real(stft), [0, 0], [FRAMES, DIM_F]).transpose(); // [2048,32]
            const im = tf.slice(tf.imag(stft), [0, 0], [FRAMES, DIM_F]).transpose();
            parts.push(re, im);
        }
        let spek = tf.stack(parts, 0).reshape([1, 4, DIM_F, FRAMES]);

        // UVRと同じく最低域3ビンをゼロ化
        spek = tf.concat([tf.zeros([1, 4, 3, FRAMES]), spek.slice([0, 0, 3, 0], [-1, -1, -1, -1])], 2);

        // onnx2tf変換後のモデルはNHWC ([1,2048,32,4])
        const nhwc = spek.transpose([0, 2, 3, 1]);
        const outputs = await runWithTfjsTensors(models[stem.key], [nhwc]);
        const pred = outputs[0].transpose([0, 3, 1, 2]).mul(COMPENSATE); // [1,4,2048,32]

        // 高域はゼロ埋めしてISTFT (2ステム版と違い元音の高域を混ぜない)
        const chReim = pred.reshape([2, 2, DIM_F, FRAMES]);
        const full = tf.concat([chReim, tf.zeros([2, 2, nBins - DIM_F, FRAMES])], 2); // [2,2,nBins,32]

        // バッチirfft: [ch, frames, nBins] の複素 → [ch, frames, nFft]
        const re = full.slice([0, 0, 0, 0], [2, 1, nBins, FRAMES]).squeeze([1]).transpose([0, 2, 1]); // [2,32,nBins]
        const im = full.slice([0, 1, 0, 0], [2, 1, nBins, FRAMES]).squeeze([1]).transpose([0, 2, 1]);
        const frames = tf.spectral.irfft(tf.complex(re, im));      // [2,32,nFft]
        const windowed = frames.mul(win);                           // 合成窓
        const frameData = await windowed.data();                    // [2*32*nFft]

        // オーバーラップ加算 + 窓の二乗和で正規化
        const outLen = SEGMENT + nFft;
        const winData = await win.data();
        const out = [new Float32Array(outLen), new Float32Array(outLen)];
        const weight = new Float32Array(outLen);
        for (let f = 0; f < FRAMES; f++) {
            const start = f * HOP;
            for (let j = 0; j < nFft; j++) weight[start + j] += winData[j] * winData[j];
        }
        for (let ch = 0; ch < 2; ch++) {
            const acc = out[ch];
            for (let f = 0; f < FRAMES; f++) {
                const src = (ch * FRAMES + f) * nFft;
                const start = f * HOP;
                for (let j = 0; j < nFft; j++) acc[start + j] += frameData[src + j];
            }
            for (let i = 0; i < outLen; i++) if (weight[i] > 1e-10) acc[i] /= weight[i];
        }
        // 中央のSEGMENT分だけ返す
        return [out[0].slice(pad, pad + SEGMENT), out[1].slice(pad, pad + SEGMENT)];
    } finally {
        tf.engine().endScope();
    }
}

// ============ 全体処理 ============
async function separate(chL, chR) {
    const total = chL.length;
    const step = SEGMENT - XFADE;
    const numSeg = Math.max(1, Math.ceil((total - XFADE) / step));
    const results = {}; // stemKey -> [Float32Array, Float32Array]
    const times = {};   // stemKey -> [ms, ...]
    for (const stem of STEMS) {
        results[stem.key] = [new Float32Array(total), new Float32Array(total)];
        times[stem.key] = [];
    }

    // 前後にコンテキストを付けてセグメントを切り出す (端はゼロ)
    function contextSlice(src, pos, pad) {
        const ctx = new Float32Array(SEGMENT + pad * 2);
        const from = pos - pad;
        const start = Math.max(0, from);
        const end = Math.min(src.length, pos + SEGMENT + pad);
        ctx.set(src.subarray(start, end), start - from);
        return ctx;
    }

    for (let s = 0; s < numSeg; s++) {
        const pos = s * step;
        for (const stem of STEMS) {
            const pad = stem.nFft / 2;
            const t0 = performance.now();
            const [outL, outR] = await processSegment(stem, contextSlice(chL, pos, pad), contextSlice(chR, pos, pad));
            times[stem.key].push(performance.now() - t0);

            // クロスフェード付きで書き込み
            const dst = results[stem.key];
            const n = Math.min(SEGMENT, total - pos);
            for (let i = 0; i < n; i++) {
                if (s > 0 && i < XFADE) {
                    const w = i / XFADE;
                    dst[0][pos + i] = dst[0][pos + i] * (1 - w) + outL[i] * w;
                    dst[1][pos + i] = dst[1][pos + i] * (1 - w) + outR[i] * w;
                } else {
                    dst[0][pos + i] = outL[i];
                    dst[1][pos + i] = outR[i];
                }
            }
        }
        setStatus(`分離中... セグメント ${s + 1}/${numSeg}`);
        await new Promise((r) => setTimeout(r, 0)); // UI更新
    }
    return { results, times };
}

// ============ WAV化・表示 ============
function toWav(chL, chR) {
    const n = chL.length;
    const buf = new ArrayBuffer(44 + n * 4);
    const v = new DataView(buf);
    const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    wstr(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); wstr(8, "WAVE");
    wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
    v.setUint32(24, SR, true); v.setUint32(28, SR * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
    wstr(36, "data"); v.setUint32(40, n * 4, true);
    for (let i = 0; i < n; i++) {
        v.setInt16(44 + i * 4, Math.max(-1, Math.min(1, chL[i])) * 32767, true);
        v.setInt16(44 + i * 4 + 2, Math.max(-1, Math.min(1, chR[i])) * 32767, true);
    }
    return new Blob([buf], { type: "audio/wav" });
}

function median(a) {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)] || 0;
}

function addPlayer(label, meta, chL, chR) {
    const div = document.createElement("div");
    div.className = "stem";
    div.innerHTML = `<b>${label}</b><span class="meta">${meta}</span>`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = URL.createObjectURL(toWav(chL, chR));
    div.appendChild(audio);
    resultsEl.appendChild(div);
}

// ============ 検証用メトリクス ============
function rms(a) {
    let s = 0;
    for (let i = 0; i < a.length; i += 13) s += a[i] * a[i];
    return Math.sqrt(s / Math.ceil(a.length / 13));
}

// 約250Hz以下の成分の割合 (一次ローパスの粗い近似)
function lowBandRatio(a) {
    let y = 0, low = 0, tot = 0;
    const k = 1 - Math.exp(-2 * Math.PI * 250 / SR);
    for (let i = 0; i < a.length; i += 3) {
        y += k * (a[i] - y);
        low += y * y;
        tot += a[i] * a[i];
    }
    return tot > 0 ? low / tot : 0;
}

// ピーク/RMS比 (パルス性の指標)
function crestFactor(a) {
    let peak = 0;
    for (let i = 0; i < a.length; i += 7) peak = Math.max(peak, Math.abs(a[i]));
    const r = rms(a);
    return r > 0 ? peak / r : 0;
}

// ============ 合成テスト音源 (自動検証用) ============
function makeSyntheticSong(seconds) {
    const n = SR * seconds;
    const L = new Float32Array(n);
    const R = new Float32Array(n);
    const bassNotes = [55, 65.4, 73.4, 82.4];
    for (let i = 0; i < n; i++) {
        const t = i / SR;
        const bar = Math.floor(t * 2) % 4;
        // ベース: 低域の持続音
        const bass = 0.3 * Math.sin(2 * Math.PI * bassNotes[bar] * t);
        // ドラム: 0.25秒ごとのノイズバースト
        const beatT = t % 0.25;
        const drum = beatT < 0.04 ? (Math.random() * 2 - 1) * 0.5 * Math.exp(-beatT * 120) : 0;
        // その他: 中域のコード
        const other = 0.12 * (Math.sin(2 * Math.PI * 440 * t) + Math.sin(2 * Math.PI * 554.4 * t) + Math.sin(2 * Math.PI * 659.3 * t));
        // ボーカル風: ビブラート付き倍音列
        const f0 = 220 * (1 + 0.015 * Math.sin(2 * Math.PI * 5.5 * t));
        const vocal = 0.18 * (Math.sin(2 * Math.PI * f0 * t) + 0.5 * Math.sin(2 * Math.PI * 2 * f0 * t) + 0.25 * Math.sin(2 * Math.PI * 3 * f0 * t));
        const v = bass + drum + other + vocal;
        L[i] = v;
        R[i] = v * 0.92 + 0.03 * Math.sin(2 * Math.PI * 330 * t);
    }
    return [L, R];
}

// ============ メイン ============
async function run(chL, chR, sourceName) {
    resultsEl.textContent = "";
    const t0 = performance.now();
    const { results, times } = await separate(chL, chR);
    const wall = (performance.now() - t0) / 1000;

    // 合算 (分離の完全性チェック用)
    const sum = [new Float32Array(chL.length), new Float32Array(chL.length)];
    for (const stem of STEMS) {
        for (let ch = 0; ch < 2; ch++) {
            const src = results[stem.key][ch];
            for (let i = 0; i < src.length; i++) sum[ch][i] += src[i];
        }
    }

    const report = { ok: true, source: sourceName, wall_seconds: +wall.toFixed(1), stems: {} };
    addPlayer("原音", sourceName, chL, chR);
    for (const stem of STEMS) {
        const [l, r] = results[stem.key];
        let finite = true;
        for (let i = 0; i < l.length; i += 97) if (!Number.isFinite(l[i])) { finite = false; break; }
        const stemRms = rms(l);
        const info = {
            median_ms: +median(times[stem.key]).toFixed(1),
            rms: +stemRms.toFixed(4),
            low_ratio: +lowBandRatio(l).toFixed(3),
            crest: +crestFactor(l).toFixed(1),
            finite,
        };
        report.stems[stem.key] = info;
        if (!finite || stemRms == 0) report.ok = false;
        addPlayer(stem.label, `n_fft=${stem.nFft} / ${info.median_ms}ms/seg / RMS ${info.rms}`, l, r);
    }
    addPlayer("4ステム合算", "分離完全性の確認用", sum[0], sum[1]);

    const perSeg = STEMS.reduce((a, s) => a + median(times[s.key]), 0);
    report.total_median_ms_per_segment = +perSeg.toFixed(1);
    // 実時間性: 1セグメント=0.72秒ぶんの音声。4モデル合計がこれを下回れば余裕
    report.realtime_factor = +((SEGMENT / SR * 1000) / perSeg).toFixed(2);

    setStatus(`完了 (${wall.toFixed(1)}秒)\n4モデル合計 ${perSeg.toFixed(0)}ms/セグメント (実時間の${report.realtime_factor}倍速)\n` +
        STEMS.map((s) => `${s.label}: ${report.stems[s.key].median_ms}ms`).join(" / "));
    return report;
}

(async function main() {
    try {
        await init();

        if (isAuto) {
            setStatus("合成テスト音源で自動検証中...");
            const [l, r] = makeSyntheticSong(8);
            const report = await run(l, r, "合成音源8秒");
            // 帯域傾向のソフトチェック (参考情報として)
            const s = report.stems;
            report.bass_is_lowest_band = s.bass.low_ratio > Math.max(s.vocals.low_ratio, s.other.low_ratio);
            report.drums_is_pulsiest = s.drums.crest >= Math.max(s.vocals.crest, s.other.crest, s.bass.crest);
            window.__stemsResult = report;
            window.__stemsDone = true;
            return;
        }

        // 手動モード: ファイル入力
        const drop = document.getElementById("drop");
        const fileInput = document.getElementById("file");
        drop.addEventListener("click", () => fileInput.click());
        drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
        drop.addEventListener("dragleave", () => drop.classList.remove("hover"));

        async function handleFile(file) {
            setStatus(`デコード中: ${file.name}`);
            const ac = new AudioContext({ sampleRate: SR });
            const buf = await ac.decodeAudioData(await file.arrayBuffer());
            ac.close();
            const l = buf.getChannelData(0);
            const r = buf.numberOfChannels > 1 ? buf.getChannelData(1) : l;
            await run(l, r, `${file.name} (${(buf.duration).toFixed(1)}秒)`);
        }
        drop.addEventListener("drop", (e) => {
            e.preventDefault();
            drop.classList.remove("hover");
            if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener("change", () => {
            if (fileInput.files[0]) handleFile(fileInput.files[0]);
        });
    } catch (e) {
        setStatus("ERROR: " + (e.stack || e));
        window.__stemsResult = { ok: false, error: String(e) };
        window.__stemsDone = true;
    }
})();
