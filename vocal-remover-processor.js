class VocalRemoverProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferL = [];
        this.bufferR = [];
        this.stopped = false;

        // メインスレッドからのメッセージ（加工済み音声）を受け取る
        this.port.onmessage = (event) => {
            if (event.data.type === "processed_buffer") {
                this.bufferL.push(...event.data.payload[0]);
                this.bufferR.push(...event.data.payload[1]);
            } else if (event.data.type === "set_stop") {
                this.stopped = event.data.value;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        if (!input || input.length === 0) return true;

        const inputL = input[0];
        const inputR = input[1] || input[0]; // モノラルの場合は左をコピー

        // 1. 入力音声をメインスレッドへ送信（AI処理用）
        this.port.postMessage({
            type: "input_data",
            payload: [Array.from(inputL), Array.from(inputR)]
        });

        // 2. 出力バッファへの書き込み
        for (let channel = 0; channel < output.length; channel++) {
            const outputData = output[channel];
            const target = (channel === 0) ? this.bufferL : this.bufferR;

            if (this.stopped) {
                // 停止時は入力をそのままスルー
                outputData.set(input[channel] || input[0]);
                continue;
            }

            if (target.length >= outputData.length) {
                // 加工済みバッファから取り出してセット
                for (let i = 0; i < outputData.length; i++) {
                    outputData[i] = target[i];
                }
                // 使用した分を削除
                target.splice(0, outputData.length);
            } else {
                // バッファ不足時は無音、または元の音を流す（アンダーフロー対策）
                outputData.set(input[channel] || input[0]);
            }
        }

        // メモリ消費抑制のため、溜まりすぎたバッファを制限（必要に応じて）
        if (this.bufferL.length > 44100) {
            this.bufferL.splice(0, this.bufferL.length - 44100);
            this.bufferR.splice(0, this.bufferR.length - 44100);
        }

        return true;
    }
}

registerProcessor("vocal-remover-processor", VocalRemoverProcessor);