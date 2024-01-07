# VocalRemoverWebExtension

## 概要
再生中のWeb動画からリアルタイムで音声のボーカルパートを除去できるChrome Extensionである。
[Tensorflow.js](https://github.com/tensorflow/tfjs)のWebGPUバックエンドを利用して推論を行っている。

## 利用上の注意
- 本拡張機能はWebGPUバックエンドを必要としますので、対応しているChrome113以降のバージョンで使用してください。
- GPUの性能が低い場合は正常に作動しない可能性があります。
- 複数の動画もしくは複数のブラウザタブで同時に使用することはできません。
- 本拡張機能は基本的に私的な利用を想定していますが、本拡張機能を利用して得られた伴奏のみの音源そのもの、もしくは加工したものを公衆に対して送信する場合は、必ず権利者の許可を得るようにしてください。

## 既知の問題
### [createScriptProcessor](https://developer.mozilla.org/ja/docs/Web/API/BaseAudioContext/createScriptProcessor)を使用している
ContentScript側で、createScriptProcessorを使用しているが、このAPIは非推奨である上UIと同じスレッドで動作するのでスクロールなどのUI操作時に音声が乱れたりする問題が起きている。
現状拡張機能からAudioWorkletを立ち上げる方法がないので、ブラウザ側の対応を待つしかない。

### コード内部のマジックナンバーが多い
申し訳ない。動くことを第一優先にPythonのコードをベースに移植したので定数定義が雑になっている箇所が多い。今後直す

### ノイズが乗ってしまう
全て0のPCM音源を入力させても、モデルで推論を行うとノイズが乗った結果が返ってくる状況を確認している。モデルを変換する上で問題を起こしている可能性があるが今後調査する。

## 今後の展望
今後は、前述の問題の解決に加えて、ボーカルと伴奏の割合を自由に調整できるようにする機能などの実装を考えている。

## 使い方
このリポジトリは基本的に開発者向けであるが、利用者の方がダウンロードしてブラウザで使いたい場合は、以下の手順で導入する。
- 緑の「Code」ボタンから「Download ZIP」を選択し、解凍する
- Chromeで、 chrome://extensions/ にアクセスする
- 右上の デベロッパー モード をオンにする
- パッケージ化されていない拡張機能を読み込む をクリックする
- 解凍した後のフォルダを選択する

## Credits
### Anjok07, Aufr33
モデルは、[MITライセンス](https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/LICENSE)に基づき、[Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)の「UVR-MDX-NET-Inst_HQ_3」を使用しています。
### Tendorflow
[Apache-2.0 license](https://github.com/tensorflow/tfjs/blob/master/LICENSE)に基づき、[Tensorflow.js](https://github.com/tensorflow/tfjs)を使用しています。

## References
[Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation"](https://arxiv.org/pdf/1706.09588.pdf)
