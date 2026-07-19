// src/tfjs_interop.ts
import { CompiledModel, getDefaultEnvironment, Tensor, TensorBufferType } from "./core.js";
import * as tf from "./tfjs-core-shim.js";
var WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE = "To convert from TFJS to LiteRT, you must use an environment that has the same WebGPU device as the TFJS backend.\n\nSince LiteRT.js sets some WebGPU options by default, the recommended way to share a WebGPU device is to let LiteRT.js create the device when it sets up the default environment and then create a new TFJS WebGPU backnend using that device:\n\n```javascript\n    const device = await getWebGpuDevice(); // from LiteRT.js\n    const adapterInfo = device.adapterInfo;\n    tf.removeBackend('webgpu');\n    tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo));\n\n    await tf.setBackend('webgpu');\n    ```";
function isWebGpuBackend() {
  return tf.getBackend().includes("webgpu");
}
var TensorConversionError = class extends Error {
  tensor;
  originalMessage;
  constructor(message) {
    super(message);
    this.originalMessage = message;
  }
  setTensor(tensor2) {
    this.message = `For tensor ${tensor2}: ${this.originalMessage}`;
    this.tensor = tensor2;
  }
};
function tfjsDtypeToLiteRt(tfjsDtype) {
  switch (tfjsDtype) {
    case "float32":
      return "float32";
    case "int32":
      return "int32";
    default:
      throw new Error(
        `Unsupported type: ${tfjsDtype}. You may need to cast to int32 or float32.`
      );
  }
}
function liteRtDtypeToTfjs(liteRtDtype) {
  switch (liteRtDtype) {
    case "float32":
      return "float32";
    case "int32":
      return "int32";
    default:
      throw new Error(`Unsupported type: ${liteRtDtype}.`);
  }
}
function tfjsToLitert(tfjsTensor, environment = getDefaultEnvironment(), preferredBufferType) {
  if (isWebGpuBackend()) {
    const backend2 = tf.backend();
    const tensorMapValue = backend2.tensorMap.get(tfjsTensor.dataId);
    const shouldReturnCpuTensor = tensorMapValue?.values && preferredBufferType === TensorBufferType.HOST_MEMORY;
    if (tensorMapValue?.resource && !shouldReturnCpuTensor) {
      if (backend2.device !== environment.webGpuDevice) {
        throw new TensorConversionError(WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE);
      }
      const gpuData = tfjsTensor.dataToGPU();
      const buffer = gpuData.buffer;
      if (!buffer) {
        throw new TensorConversionError(
          "TFJS tensor did not have a GPU buffer."
        );
      }
      const dtype = tfjsDtypeToLiteRt(tfjsTensor.dtype);
      return new Tensor(buffer, tfjsTensor.shape, dtype, environment, () => {
        gpuData.tensorRef.dispose();
      });
    }
  }
  const tfjsData = tfjsTensor.dataSync();
  return new Tensor(tfjsData, tfjsTensor.shape, environment);
}
function litertToTfjs(tensor2) {
  switch (tensor2.bufferType) {
    case TensorBufferType.HOST_MEMORY:
      return litertToTfjsCpu(tensor2);
    case TensorBufferType.WEB_GPU_BUFFER_PACKED:
      return litertToTfjsWebGpu(tensor2);
    default:
      throw new Error("Unsupported accelerator: " + tensor2.accelerator);
  }
}
function litertToTfjsCpu(tensor2) {
  const typedArray = tensor2.toTypedArray();
  const tfjsDataType = liteRtDtypeToTfjs(tensor2.type.dtype);
  return tf.tensor(
    typedArray,
    [...tensor2.type.layout.dimensions],
    tfjsDataType
  );
}
function litertToTfjsWebGpu(tensor2) {
  if (!isWebGpuBackend()) {
    throw new TensorConversionError(
      `LiteRT WebGPU tensors can only be converted to TFJS WebGPU tensors, but the TFJS backend is ${tf.getBackend()}. If you want to convert to a CPU TFJS tensor, please first move (or copy) the LiteRT tensor to CPU with \`tensor.moveTo('wasm')\` (or \`tensor.copyTo('wasm')\`).`
    );
  }
  const backend2 = tf.backend();
  const device = backend2.device;
  if (device !== tensor2.environment.webGpuDevice) {
    throw new TensorConversionError(WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE);
  }
  const litertBuffer = tensor2.toGpuBuffer();
  const requiredSizeInBytes = tensor2.liteRtTensorBuffer.size();
  let buffer;
  if (litertBuffer.size === requiredSizeInBytes) {
    buffer = litertBuffer;
  } else {
    buffer = device.createBuffer({
      size: requiredSizeInBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      litertBuffer,
      0,
      buffer,
      0,
      requiredSizeInBytes
    );
    device.queue.submit([commandEncoder.finish()]);
  }
  const tfjsDataType = liteRtDtypeToTfjs(tensor2.type.dtype);
  return tf.tensor({ buffer }, [...tensor2.type.layout.dimensions], tfjsDataType);
}
function mapOnContainer(t, f, isA) {
  if (isA(t)) {
    return f(t);
  } else if (Array.isArray(t)) {
    return t.map(f);
  } else {
    return Object.fromEntries(Object.entries(t).map(([key, a]) => [key, f(a, key)]));
  }
}
function mapOnTfjs(t, f) {
  return mapOnContainer(t, f, (val) => val instanceof tf.Tensor);
}
function mapOnLiteRt(t, f) {
  return mapOnContainer(t, f, (val) => val instanceof Tensor);
}
async function runWithTfjsTensors(model, inputOrSignatureName, maybeInputOrLiteRt) {
  let signature;
  let tfjsInputs;
  if (typeof inputOrSignatureName === "string") {
    signature = inputOrSignatureName;
    tfjsInputs = maybeInputOrLiteRt;
  } else {
    tfjsInputs = inputOrSignatureName;
  }
  const litertInputDetails = model.getInputDetails();
  const inputs = mapOnTfjs(tfjsInputs, (tfjsTensor, keyOrIndex) => {
    try {
      let inputDetails;
      if (typeof keyOrIndex === "number") {
        inputDetails = litertInputDetails[keyOrIndex];
      } else if (typeof keyOrIndex === "string") {
        inputDetails = litertInputDetails.find((details) => details.name === keyOrIndex);
      }
      const preferredBufferType = inputDetails?.supportedBufferTypes.values().next().value;
      return tfjsToLitert(
        tfjsTensor,
        model.options.environment,
        preferredBufferType
      );
    } catch (e) {
      if (e instanceof TensorConversionError && keyOrIndex !== void 0) {
        e.setTensor(keyOrIndex);
      }
      throw e;
    }
  });
  let outputs;
  if (signature) {
    if (!(model instanceof CompiledModel)) {
      throw new Error(
        "Signature specified but a SignatureRunner was passed instead of a model"
      );
    }
    outputs = await model.run(signature, inputs);
  } else {
    outputs = await model.run(inputs);
  }
  mapOnLiteRt(inputs, (tensor2) => {
    tensor2.delete();
  });
  return mapOnLiteRt(outputs, (tensor2, keyOrIndex) => {
    let tfjsTensor;
    try {
      tfjsTensor = litertToTfjs(tensor2);
    } catch (e) {
      if (e instanceof TensorConversionError && keyOrIndex !== void 0) {
        e.setTensor(keyOrIndex);
      }
      throw e;
    } finally {
      tensor2.delete();
    }
    return tfjsTensor;
  });
}
export {
  TensorConversionError,
  litertToTfjs,
  runWithTfjsTensors,
  tfjsToLitert
};
//# sourceMappingURL=index.js.map