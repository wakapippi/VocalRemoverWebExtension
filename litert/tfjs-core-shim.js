// tfjs-interop.js が import する "@tensorflow/tfjs-core" を、
// UMD版 (tf.min.js) のグローバル tf にリダイレクトするシム。
// (sandboxページはCSPでインラインscriptが使えず importmap を書けないため、
//  ベンダリング時にimportパスをこのファイルへ書き換えている)
const tf = globalThis.tf;

export const Tensor = tf.Tensor;
export const backend = (...args) => tf.backend(...args);
export const getBackend = (...args) => tf.getBackend(...args);
export const registerBackend = (...args) => tf.registerBackend(...args);
export const removeBackend = (...args) => tf.removeBackend(...args);
export const setBackend = (...args) => tf.setBackend(...args);
export const tensor = (...args) => tf.tensor(...args);
