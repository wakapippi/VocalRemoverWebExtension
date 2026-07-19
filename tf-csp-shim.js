// tf.min.js のCSP対応シム。tf.min.js 内の regenerator-runtime は
//   try { regeneratorRuntime = t } catch (e) { Function("r", "regeneratorRuntime = r")(t) }
// というフォールバックを持ち、strictモードでは未宣言グローバルへの代入が throw して
// catch側の Function() がCSP ('unsafe-eval'なし) でブロックされ、tf全体の初期化が壊れる。
// 事前にグローバル変数を宣言しておくことで try 側の代入を成功させ、evalを回避する。
var regeneratorRuntime;
