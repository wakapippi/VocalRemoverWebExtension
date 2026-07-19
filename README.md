# VocalRemoverWebExtension

## 概要
再生中のWeb動画からリアルタイムで音声のボーカルパートを除去できるChrome Extensionである。
[Tensorflow.js](https://github.com/tensorflow/tfjs)のWebGPUバックエンドを利用して推論を行っている。

## 利用上の注意
- 本拡張機能はWebGPUバックエンドを必要としますので、対応しているChrome113以降のバージョンで使用してください。
- GPUの性能が低い場合は正常に作動しない可能性があります。
- 複数の動画もしくは複数のブラウザタブで同時に使用することはできません。
- 本拡張機能は基本的に私的な利用を想定していますが、本拡張機能を利用して得られた伴奏のみの音源そのもの、もしくは加工したものを公衆に対して送信する場合は、必ず権利者の許可を得るようにしてください。

## 機能
- 再生中のWeb動画からリアルタイムでボーカルを除去
- ボーカル / 伴奏それぞれの音量をスライダーで調整可能 (ボーカルのみの再生も可能)
- ノイズ除去オプション (MDXモデル特有のノイズフロアを、極性反転入力との平均で相殺する。GPU負荷は約2倍)
- 無音区間 (一時停止中など) は推論を自動的にスキップしてGPU負荷を抑制

## 既知の問題
### 分離品質がUVR本体より低い
リアルタイム処理のためにモデルの時間フレーム数を256から32に縮めて変換しており、モデルが参照できる文脈が短いぶん分離品質は本家より劣る。

### 複数タブでの同時使用不可
複数の動画もしくは複数のブラウザタブで同時に使用することはできない。

## 今後の展望
分離品質の改善 (より長いセグメントでの推論とレイテンシのバランス調整) などを検討している。

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
