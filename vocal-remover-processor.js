const INPUT_BATCH_SIZE = 2048;   // 入力をまとめて送るサンプル数 (約46ms @ 44.1kHz)
const MAX_QUEUE_SIZE = 44100;    // 処理済みバッファの上限 (約1秒)

class VocalRemoverProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.stopped = false;

        // 入力バッチ (メインスレッドへの送信頻度を約3ms間隔→約46ms間隔に下げる)
        this.batchL = new Float32Array(INPUT_BATCH_SIZE);
        this.batchR = new Float32Array(INPUT_BATCH_SIZE);
        this.batchFill = 0;

        // 処理済み音声のチャンクキュー (spread/spliceによる全体コピーを避ける)
        this.chunks = [];
        this.chunkOffset = 0; // 先頭チャンクの読み出し位置
        this.queued = 0;      // キュー内の残りサンプル数

        // アンダーフロー時のフェード制御用
        this.everProcessed = false; // 処理済み音声を一度でも出力したか
        this.wasUnderflow = true;   // 直前のフレームがアンダーフローだったか
        this.lastL = 0;             // 最後に出力したサンプル値 (フェードアウト用)
        this.lastR = 0;

        this.port.onmessage = (event) => {
            if (event.data.type === "processed_buffer") {
                this.chunks.push(event.data.payload);
                this.queued += event.data.payload[0].length;
                this.trimQueue();
            } else if (event.data.type === "set_stop") {
                this.stopped = event.data.value;
                if (!this.stopped) {
                    // 再開時は古い処理済み音声を破棄
                    this.chunks = [];
                    this.chunkOffset = 0;
                    this.queued = 0;
                    this.everProcessed = false;
                    this.wasUnderflow = true;
                }
            }
        };
    }

    // 溜まりすぎたバッファを古い方から破棄
    trimQueue() {
        while (this.queued > MAX_QUEUE_SIZE && this.chunks.length > 0) {
            const first = this.chunks[0];
            const available = first[0].length - this.chunkOffset;
            const drop = Math.min(available, this.queued - MAX_QUEUE_SIZE);
            this.chunkOffset += drop;
            this.queued -= drop;
            if (this.chunkOffset >= first[0].length) {
                this.chunks.shift();
                this.chunkOffset = 0;
            }
        }
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        if (!input || input.length === 0) return true;

        const inputL = input[0];
        const inputR = input[1] || input[0]; // モノラルの場合は左をコピー

        // 1. 入力をバッチに蓄積し、満杯になったらTransferableでゼロコピー送信
        this.batchL.set(inputL, this.batchFill);
        this.batchR.set(inputR, this.batchFill);
        this.batchFill += inputL.length;
        if (this.batchFill >= INPUT_BATCH_SIZE) {
            const l = this.batchL;
            const r = this.batchR;
            this.port.postMessage({ type: "input_data", payload: [l, r] }, [l.buffer, r.buffer]);
            this.batchL = new Float32Array(INPUT_BATCH_SIZE);
            this.batchR = new Float32Array(INPUT_BATCH_SIZE);
            this.batchFill = 0;
        }

        // 2. 出力バッファへの書き込み
        const frameSize = output[0].length;

        if (this.stopped) {
            // 停止中は入力をそのままスルー
            for (let channel = 0; channel < output.length; channel++) {
                output[channel].set(input[channel] || input[0]);
            }
            this.wasUnderflow = true;
            return true;
        }

        if (this.queued >= frameSize) {
            // 処理済みバッファから取り出してセット
            for (let i = 0; i < frameSize; i++) {
                const first = this.chunks[0];
                output[0][i] = first[0][this.chunkOffset];
                if (output[1]) output[1][i] = first[1][this.chunkOffset];
                this.chunkOffset++;
                if (this.chunkOffset >= first[0].length) {
                    this.chunks.shift();
                    this.chunkOffset = 0;
                }
            }
            this.queued -= frameSize;

            if (this.wasUnderflow) {
                // 無音からの復帰時はフェードインでクリック音を防ぐ
                for (let i = 0; i < frameSize; i++) {
                    const g = i / frameSize;
                    output[0][i] *= g;
                    if (output[1]) output[1][i] *= g;
                }
                this.wasUnderflow = false;
            }
            this.everProcessed = true;
            this.lastL = output[0][frameSize - 1];
            this.lastR = output[1] ? output[1][frameSize - 1] : this.lastL;
        } else if (!this.everProcessed) {
            // 起動直後で処理済み音声がまだ届いていない間は元音をそのまま流す
            for (let channel = 0; channel < output.length; channel++) {
                output[channel].set(input[channel] || input[0]);
            }
        } else if (!this.wasUnderflow) {
            // 処理開始後のバッファ不足時は元音 (ボーカル入り) を流さず、
            // 最後のサンプル値からフェードアウトした無音にする
            for (let i = 0; i < frameSize; i++) {
                const g = 1 - i / frameSize;
                output[0][i] = this.lastL * g;
                if (output[1]) output[1][i] = this.lastR * g;
            }
            this.wasUnderflow = true;
        }
        // (上記以外 = アンダーフロー継続中は、ゼロ初期化済みの出力バッファのまま無音)

        return true;
    }
}

registerProcessor("vocal-remover-processor", VocalRemoverProcessor);
