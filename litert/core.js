// src/wasm_binding_types.ts
var ElementType = {
  NONE: 0,
  FLOAT32: 1,
  INT32: 2,
  UINT8: 3,
  INT64: 4,
  STRING: 5,
  BOOL: 6,
  INT16: 7,
  COMPLEX64: 8,
  INT8: 9,
  FLOAT16: 10,
  FLOAT64: 11,
  COMPLEX128: 12,
  UINT64: 13,
  RESOURCE: 14,
  VARIANT: 15,
  UINT32: 16,
  UINT16: 17,
  INT4: 18,
  BFLOAT16: 19
};
var ElementTypeName = {
  [ElementType.NONE]: "NONE",
  [ElementType.FLOAT32]: "FLOAT32",
  [ElementType.INT32]: "INT32",
  [ElementType.UINT8]: "UINT8",
  [ElementType.INT64]: "INT64",
  [ElementType.STRING]: "STRING",
  [ElementType.BOOL]: "BOOL",
  [ElementType.INT16]: "INT16",
  [ElementType.COMPLEX64]: "COMPLEX64",
  [ElementType.INT8]: "INT8",
  [ElementType.FLOAT16]: "FLOAT16",
  [ElementType.FLOAT64]: "FLOAT64",
  [ElementType.COMPLEX128]: "COMPLEX128",
  [ElementType.UINT64]: "UINT64",
  [ElementType.RESOURCE]: "RESOURCE",
  [ElementType.VARIANT]: "VARIANT",
  [ElementType.UINT32]: "UINT32",
  [ElementType.UINT16]: "UINT16",
  [ElementType.INT4]: "INT4",
  [ElementType.BFLOAT16]: "BFLOAT16"
};
var TensorBufferType = {
  HOST_MEMORY: 1,
  WEB_GPU_BUFFER: 20,
  WEB_GPU_BUFFER_FP16: 21,
  WEB_GPU_BUFFER_PACKED: 26
};
var TensorBufferTypeName = {
  [TensorBufferType.HOST_MEMORY]: "HOST_MEMORY",
  [TensorBufferType.WEB_GPU_BUFFER]: "WEB_GPU_BUFFER",
  [TensorBufferType.WEB_GPU_BUFFER_FP16]: "WEB_GPU_BUFFER_FP16",
  [TensorBufferType.WEB_GPU_BUFFER_PACKED]: "WEB_GPU_BUFFER_PACKED"
};

// src/datatypes.ts
var DATATYPES = Object.freeze([
  {
    dtype: "float32",
    typedArrayConstructor: Float32Array,
    elementType: ElementType.FLOAT32
  },
  {
    dtype: "int32",
    typedArrayConstructor: Int32Array,
    elementType: ElementType.INT32
  },
  {
    dtype: "uint8",
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  }
]);
function getDataType(val) {
  for (const dataTypeMapping of DATATYPES) {
    if (dataTypeMapping.dtype === val || dataTypeMapping.typedArrayConstructor === val || val instanceof dataTypeMapping.typedArrayConstructor || dataTypeMapping.elementType === val) {
      return dataTypeMapping;
    }
  }
  if (typeof val === "string") {
    throw new Error(`DType ${val} is not supported.`);
  } else if (val instanceof Object) {
    throw new Error(`Typed array ${"name" in val ? val.name : val.constructor.name} is not supported.`);
  } else {
    throw new Error(
      `Element type ${ElementTypeName[val] ?? val} is not supported.`
    );
  }
}

// src/global_litert.ts
var LiteRtNotLoadedError = class extends Error {
  constructor() {
    super(
      "LiteRT is not initialized yet. Please call loadLiteRt() and wait for its promise to resolve to load the LiteRT WASM module."
    );
  }
};
var globalLiteRt = void 0;
var globalLiteRtPromise = void 0;
function getGlobalLiteRt() {
  if (!globalLiteRt) {
    throw new LiteRtNotLoadedError();
  }
  return globalLiteRt;
}
function hasGlobalLiteRt() {
  return Boolean(globalLiteRt);
}
function setGlobalLiteRt(liteRt) {
  globalLiteRt = liteRt;
}
function getGlobalLiteRtPromise() {
  return globalLiteRtPromise;
}
function hasGlobalLiteRtPromise() {
  return Boolean(globalLiteRtPromise);
}
function setGlobalLiteRtPromise(promise) {
  globalLiteRtPromise = promise;
}

// src/accelerator_types.ts
var AcceleratorDefaultTensorBufferType = {
  "webgpu": TensorBufferType.WEB_GPU_BUFFER_PACKED,
  "wasm": TensorBufferType.HOST_MEMORY
};
var TensorBufferTypeToAccelerator = {
  [TensorBufferType.HOST_MEMORY]: "wasm",
  [TensorBufferType.WEB_GPU_BUFFER]: "webgpu",
  [TensorBufferType.WEB_GPU_BUFFER_FP16]: "webgpu",
  [TensorBufferType.WEB_GPU_BUFFER_PACKED]: "webgpu"
};

// src/environment.ts
var DESIRED_WEBGPU_FEATURES = [
  "shader-f16",
  "subgroups"
];
var Environment = class _Environment {
  constructor(options) {
    this.options = options;
    this.liteRtEnvironment = getGlobalLiteRt().liteRtWasm.LiteRtEnvironment.create(
      options.webGpuDevice
    );
  }
  liteRtEnvironment;
  static async create(options = {}) {
    let webGpuDevice = null;
    if ("webGpuDevice" in options) {
      if (options.webGpuDevice) {
        webGpuDevice = options.webGpuDevice;
      }
    } else {
      try {
        webGpuDevice = await createDefaultWebGpuDevice();
      } catch (e) {
        console.warn("Failed to create default WebGPU device:", e);
      }
    }
    return new _Environment({
      ...options,
      webGpuDevice
    });
  }
  get webGpuDevice() {
    return this.options.webGpuDevice;
  }
  delete() {
    this.liteRtEnvironment.delete();
  }
};
async function createDefaultWebGpuDevice() {
  const adapterDescriptor = {
    powerPreference: "high-performance"
  };
  const adapter = await navigator.gpu.requestAdapter(adapterDescriptor);
  if (!adapter) {
    throw new Error("No GPU adapter found.");
  }
  const requiredLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    maxTextureDimension2D: adapter.limits.maxTextureDimension2D
  };
  const requiredFeatures = [];
  for (const feature of DESIRED_WEBGPU_FEATURES) {
    if (adapter.features.has(feature)) {
      requiredFeatures.push(feature);
    }
  }
  return await adapter.requestDevice({
    requiredFeatures,
    requiredLimits
  });
}

// src/wasm_utils.ts
function emscriptenVectorToArray(vector) {
  const array = new Array(vector.size());
  for (let i = 0; i < vector.size(); ++i) {
    array[i] = vector.get(i);
  }
  vector.delete();
  return array;
}
function fillEmscriptenVector(data, vector) {
  for (const item of data) {
    vector.push_back(item);
  }
}

// src/tensor.ts
function parseData(remainingArgs) {
  const data = remainingArgs.shift();
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  if (data instanceof liteRtWasm.LiteRtTensorBuffer) {
    return { liteRtTensorBuffer: data };
  } else if (ArrayBuffer.isView(data)) {
    return { typedArray: data };
  } else if (data instanceof GPUBuffer) {
    return { gpuBuffer: data };
  } else {
    throw new Error(
      `Unknown type (${data?.constructor.name ?? data}) provided to create a Tensor`
    );
  }
}
function parseShape(remainingArgs) {
  if (Array.isArray(remainingArgs[0]) || remainingArgs[0] instanceof Int32Array) {
    return { shape: remainingArgs.shift() };
  } else {
    return {};
  }
}
function shiftUntilDefined(remainingArgs) {
  while (remainingArgs.length > 0 && remainingArgs[0] === void 0) {
    remainingArgs.shift();
  }
}
function parseDataType(remainingArgs) {
  shiftUntilDefined(remainingArgs);
  if (typeof remainingArgs[0] === "string") {
    const dtype = remainingArgs.shift();
    return { dataType: getDataType(dtype).dtype };
  } else {
    return {};
  }
}
function parseEnvironment(remainingArgs) {
  shiftUntilDefined(remainingArgs);
  if (remainingArgs[0] instanceof Environment) {
    return { environment: remainingArgs.shift() };
  } else {
    return {};
  }
}
function parseOnDelete(remainingArgs) {
  shiftUntilDefined(remainingArgs);
  if (remainingArgs[0] instanceof Function) {
    return { onDelete: remainingArgs.shift() };
  } else {
    return {};
  }
}
function parseArgs(args) {
  return {
    ...parseData(args),
    ...parseShape(args),
    ...parseDataType(args),
    ...parseEnvironment(args),
    ...parseOnDelete(args)
  };
}
var Tensor = class _Tensor {
  liteRtTensorBuffer;
  type;
  environment;
  deletedInternal = false;
  onDelete;
  static copyFunctions = /* @__PURE__ */ new Map();
  constructor(a, b, c, d, e) {
    const {
      typedArray,
      gpuBuffer,
      liteRtTensorBuffer,
      shape,
      dataType,
      environment,
      onDelete
    } = parseArgs([a, b, c, d, e]);
    this.onDelete = onDelete;
    this.environment = environment ?? getGlobalLiteRt().getDefaultEnvironment();
    if (liteRtTensorBuffer) {
      if (shape) {
        throw new Error(
          "A LiteRtTensorBuffer cannot be provided with a shape."
        );
      }
      if (dataType) {
        throw new Error(
          "A LiteRtTensorBuffer cannot be provided with a data type."
        );
      }
      this.liteRtTensorBuffer = liteRtTensorBuffer;
    } else if (gpuBuffer) {
      if (!shape) {
        throw new Error("A GPUBuffer must be provided with a shape.");
      }
      if (!dataType) {
        throw new Error("A GPUBuffer must be provided with a data type.");
      }
      const [liteRtTensorBuffer2, webGpuBufferPtr] = webGpuBufferToLiteRtTensorBuffer(
        gpuBuffer,
        shape,
        dataType,
        this.environment
      );
      this.liteRtTensorBuffer = liteRtTensorBuffer2;
      const onDelete2 = this.onDelete;
      this.onDelete = () => {
        const liteRtWasm = getGlobalLiteRt().liteRtWasm;
        liteRtWasm.wgpuBufferRelease(webGpuBufferPtr);
        onDelete2?.();
      };
    } else if (typedArray) {
      this.liteRtTensorBuffer = typedArrayToLiteRtTensorBuffer(
        typedArray,
        shape,
        environment
      );
    } else {
      throw new Error("No data provided to create a Tensor.");
    }
    this.type = liteRtTensorBufferToTensorType(this.liteRtTensorBuffer);
  }
  static fromTypedArray(data, shape, environment) {
    return new _Tensor(data, shape, environment);
  }
  ensureNotDeleted() {
    if (this.deleted) {
      throw new Error("Tensor is deleted and cannot be used.");
    }
  }
  async data() {
    this.ensureNotDeleted();
    if (this.liteRtTensorBuffer.bufferType().value === TensorBufferType.HOST_MEMORY) {
      return this.toTypedArray();
    }
    const copy = await this.copyTo("wasm");
    const data = await copy.data();
    copy.delete();
    return data;
  }
  toTypedArray() {
    this.ensureNotDeleted();
    const liteRtWasm = getGlobalLiteRt().liteRtWasm;
    if (this.liteRtTensorBuffer.isWebGpuMemory()) {
      throw new Error(
        "Cannot convert a Tensor with WebGPU memory to a TypedArray."
      );
    }
    if (this.liteRtTensorBuffer.bufferType().value !== liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY.value) {
      throw new Error(
        "Cannot convert a Tensor with non-host memory to a TypedArray."
      );
    }
    if (this.liteRtTensorBuffer.size() !== this.liteRtTensorBuffer.packedSize() || this.liteRtTensorBuffer.offset() !== 0) {
      throw new Error("Tensors with strides or padding are not yet supported.");
    }
    const rankedTensorType = this.liteRtTensorBuffer.tensorType();
    const elementType = rankedTensorType.elementType();
    const byteWidth = liteRtWasm.liteRtGetByteWidth(elementType);
    rankedTensorType.delete();
    const typedArrayConstructor = getDataType(
      elementType.value
    ).typedArrayConstructor;
    if (typedArrayConstructor.BYTES_PER_ELEMENT !== byteWidth) {
      throw new Error(
        `Byte width ${byteWidth} of the tensor's element type ${ElementTypeName[elementType.value]} does not match the expected byte width ${typedArrayConstructor.BYTES_PER_ELEMENT} of the ${typedArrayConstructor.name}.`
      );
    }
    const dataPtr = this.liteRtTensorBuffer.lock(
      getGlobalLiteRt().liteRtWasm.LiteRtTensorBufferLockMode.READ
    );
    try {
      const uint8Array = liteRtWasm.HEAPU8.slice(
        dataPtr,
        dataPtr + this.liteRtTensorBuffer.packedSize()
      );
      const typedArray = new typedArrayConstructor(
        uint8Array.buffer,
        uint8Array.byteOffset,
        uint8Array.byteLength / byteWidth
      );
      return typedArray;
    } finally {
      this.liteRtTensorBuffer.unlock();
    }
  }
  getBufferType() {
    this.ensureNotDeleted();
    return this.liteRtTensorBuffer.bufferType().value;
  }
  /**
   * Returns the underlying GPUBuffer of the Tensor.
   *
   * Note that the lifetime of the returned GPUBuffer is dependant upon how the
   * Tensor was created. If the Tensor was constructed from a GPUBuffer, then
   * the GPUBuffer will NOT be released when the Tensor is deleted. If the
   * Tensor was copied/moved to GPU from host memory, then the GPU buffer will
   * be released when the Tensor is deleted.
   *
   * The GPU buffer may be larger than the actual data in the tensor.
   *
   * @return The GPUBuffer containing the Tensor's data.
   */
  toGpuBuffer() {
    this.ensureNotDeleted();
    const liteRtWasm = getGlobalLiteRt().liteRtWasm;
    if (!this.liteRtTensorBuffer.isWebGpuMemory()) {
      throw new Error(
        "Cannot convert a Tensor with non-WebGPU memory to a GPUBuffer."
      );
    }
    const bufferTypeValue = this.liteRtTensorBuffer.bufferType().value;
    if (bufferTypeValue !== liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER.value && bufferTypeValue !== liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_FP16.value && bufferTypeValue !== liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED.value) {
      throw new Error(
        "Cannot convert a Tensor with host memory to a GPUBuffer."
      );
    }
    if (this.liteRtTensorBuffer.size() !== this.liteRtTensorBuffer.packedSize() || this.liteRtTensorBuffer.offset() !== 0) {
      throw new Error("Tensors with strides or padding are not yet supported.");
    }
    const gpuBufferId = this.liteRtTensorBuffer.getWebGpuBuffer();
    return liteRtWasm.WebGPU.getJsObject(gpuBufferId);
  }
  getCopyFunctionSet(destination) {
    this.ensureNotDeleted();
    const sourceBufferType = this.getBufferType();
    const copyFunctions = _Tensor.copyFunctions.get(sourceBufferType);
    if (!copyFunctions) {
      throw new Error(
        `TensorBufferType ${TensorBufferTypeName[sourceBufferType] ?? sourceBufferType} does not support copying or moving`
      );
    }
    const destinationBufferType = typeof destination === "string" ? AcceleratorDefaultTensorBufferType[destination] : destination;
    if (destinationBufferType == null) {
      throw new Error(
        `Unknown destination '${destination}' for copying or moving.`
      );
    }
    const copyFunctionSet = copyFunctions.get(destinationBufferType);
    if (!copyFunctionSet) {
      const supportedDestinations = [...copyFunctions].map(
        ([key]) => TensorBufferTypeName[key] ?? key
      );
      throw new Error(
        `TensorBufferType ${TensorBufferTypeName[sourceBufferType]} does not support copying or moving to ${TensorBufferTypeName[destinationBufferType]}. It supports the following TensorBufferTypes: [${supportedDestinations.join(
          ", "
        )}].`
      );
    }
    return [copyFunctionSet, destinationBufferType];
  }
  /**
   * Copies the tensor to the given accelerator.
   *
   * @param destination The accelerator or buffer type to copy to.
   * @return A promise that resolves to the copied tensor.
   */
  async copyTo(destination, options) {
    const [copyFunctionSet, destinationBufferType] = this.getCopyFunctionSet(destination);
    if (!copyFunctionSet.copyTo) {
      throw new Error(
        `Copying to ${TensorBufferTypeName[destinationBufferType]} is not supported by this tensor.`
      );
    }
    return copyFunctionSet.copyTo(this, options);
  }
  /**
   * Moves the tensor to the given accelerator.
   *
   * @param destination The accelerator or buffer type to move to.
   * @return A promise that resolves to the moved tensor.
   */
  async moveTo(destination, options) {
    const [copyFunctionSet, destinationBufferType] = this.getCopyFunctionSet(destination);
    if (!copyFunctionSet.moveTo) {
      throw new Error(
        `Moving to ${TensorBufferTypeName[destinationBufferType]} is not supported by this tensor.`
      );
    }
    return copyFunctionSet.moveTo(this, options);
  }
  get bufferType() {
    return this.liteRtTensorBuffer.bufferType().value;
  }
  get accelerator() {
    const accelerator = TensorBufferTypeToAccelerator[this.bufferType];
    if (accelerator === void 0) {
      throw new Error(
        `TensorBufferType ${TensorBufferTypeName[this.bufferType]} has an unknown accelerator type.`
      );
    }
    return accelerator;
  }
  get deleted() {
    return this.deletedInternal;
  }
  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtTensorBuffer.delete();
    this.onDelete?.();
  }
};
function liteRtTensorBufferToTensorType(liteRtTensorBuffer) {
  const liteRtRankedTensorType = liteRtTensorBuffer.tensorType();
  const elementType = liteRtRankedTensorType.elementType();
  const liteRtLayout = liteRtRankedTensorType.layout();
  const dimensions = liteRtLayout.dimensions();
  liteRtLayout.delete();
  liteRtRankedTensorType.delete();
  return {
    dtype: getDataType(elementType.value).dtype,
    layout: { dimensions: emscriptenVectorToArray(dimensions) }
  };
}
function webGpuBufferToLiteRtTensorBuffer(gpuBuffer, shape, dtype, environment) {
  const globalLiteRt2 = getGlobalLiteRt();
  const liteRtWasm = globalLiteRt2.liteRtWasm;
  const dimensionsVector = new liteRtWasm.VectorInt32();
  fillEmscriptenVector(shape, dimensionsVector);
  const layout = liteRtWasm.LiteRtLayout.create(dimensionsVector);
  dimensionsVector.delete();
  const rankedTensorType = liteRtWasm.LiteRtRankedTensorType.create(
    { value: getDataType(dtype).elementType },
    layout
  );
  layout.delete();
  const importedGpuBufferPtr = liteRtWasm.WebGPU.importJsBuffer(gpuBuffer);
  const liteRtTensorBuffer = liteRtWasm.LiteRtTensorBuffer.createFromWebGpuBuffer(
    environment.liteRtEnvironment,
    rankedTensorType,
    liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED,
    importedGpuBufferPtr,
    gpuBuffer.size
  );
  rankedTensorType.delete();
  return [liteRtTensorBuffer, importedGpuBufferPtr];
}
function typedArrayToLiteRtTensorBuffer(data, shape, environment) {
  const globalLiteRt2 = getGlobalLiteRt();
  const liteRtWasm = globalLiteRt2.liteRtWasm;
  environment = environment ?? globalLiteRt2.getDefaultEnvironment();
  const elementType = getDataType(data).elementType;
  const dimensionsVector = new liteRtWasm.VectorInt32();
  fillEmscriptenVector(shape ?? [data.length], dimensionsVector);
  const layout = liteRtWasm.LiteRtLayout.create(dimensionsVector);
  dimensionsVector.delete();
  const expectedNumElements = layout.numElements();
  if (data.length !== expectedNumElements) {
    layout.delete();
    throw new Error(
      `Number of elements ${data.length} of the provided TypedArray does not match the expected number of elements ${expectedNumElements}.`
    );
  }
  const rankedTensorType = liteRtWasm.LiteRtRankedTensorType.create(
    { value: elementType },
    layout
  );
  layout.delete();
  const arrayType = data.constructor;
  const bufferSize = arrayType.BYTES_PER_ELEMENT * data.length;
  const expectedBufferSize = rankedTensorType.bytes();
  if (bufferSize !== expectedBufferSize) {
    rankedTensorType.delete();
    throw new Error(
      `Byte length ${bufferSize} of the provided TypedArray does not match the expected buffer size ${expectedBufferSize}.`
    );
  }
  const liteRtTensorBuffer = liteRtWasm.LiteRtTensorBuffer.createManaged(
    environment.liteRtEnvironment,
    liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY,
    rankedTensorType,
    bufferSize
  );
  rankedTensorType.delete();
  const dataPtr = liteRtTensorBuffer.lock(
    liteRtWasm.LiteRtTensorBufferLockMode.WRITE
  );
  try {
    const uint8Data = new Uint8Array(
      data.buffer,
      data.byteOffset,
      data.byteLength
    );
    liteRtWasm.HEAPU8.set(uint8Data, dataPtr);
  } finally {
    liteRtTensorBuffer.unlock();
  }
  return liteRtTensorBuffer;
}

// src/signature_runner.ts
var CompiledModelSignatureRunner = class {
  constructor(signatureIndex, liteRtModel, liteRtCompiledModel, options) {
    this.signatureIndex = signatureIndex;
    this.liteRtModel = liteRtModel;
    this.liteRtCompiledModel = liteRtCompiledModel;
    this.options = options;
    this.liteRtSimpleSignature = liteRtModel.getSignature(signatureIndex);
    const inputNames = emscriptenVectorToArray(this.liteRtSimpleSignature.inputNames());
    const inputDetails = [];
    for (let i = 0; i < inputNames.length; i++) {
      const name = inputNames[i];
      const tensorType = liteRtModel.getInputTensorType(signatureIndex, i);
      const requirements = liteRtCompiledModel.getInputBufferRequirements(signatureIndex, i);
      inputDetails.push(makeTensorDetails(name, i, tensorType, requirements));
    }
    this.inputDetails = Object.freeze(inputDetails);
    const outputNames = emscriptenVectorToArray(this.liteRtSimpleSignature.outputNames());
    const outputDetails = [];
    for (let i = 0; i < outputNames.length; i++) {
      const name = outputNames[i];
      const tensorType = liteRtModel.getOutputTensorType(signatureIndex, i);
      const requirements = liteRtCompiledModel.getOutputBufferRequirements(signatureIndex, i);
      outputDetails.push(makeTensorDetails(name, i, tensorType, requirements));
    }
    this.outputDetails = Object.freeze(outputDetails);
  }
  inputDetails;
  outputDetails;
  liteRtSimpleSignature;
  deletedInternal = false;
  /**
   * The string key corresponding to this signature in the model.
   */
  get key() {
    this.ensureNotDeleted();
    return this.liteRtSimpleSignature.key();
  }
  /**
   * Get details about each input tensor.
   */
  getInputDetails() {
    this.ensureNotDeleted();
    return this.inputDetails;
  }
  /**
   * Get details about each output tensor.
   */
  getOutputDetails() {
    this.ensureNotDeleted();
    return this.outputDetails;
  }
  async run(input) {
    this.ensureNotDeleted();
    const inputArray = this.inputsToArray(input);
    const { inputsOnAccelerator, cleanup } = await this.ensureInputsOnAccelerator(inputArray);
    let outputArray;
    try {
      outputArray = await this.runWithArray(inputsOnAccelerator);
    } finally {
      cleanup();
    }
    if (Array.isArray(input) || input instanceof Tensor) {
      return outputArray;
    } else {
      return this.outputsToRecord(outputArray);
    }
  }
  inputsToArray(input) {
    if (Array.isArray(input)) {
      if (input.length !== this.inputDetails.length) {
        throw new Error(
          `run() called with ${input.length} inputs, but signature expects ${this.inputDetails.length} inputs`
        );
      }
      return input;
    }
    if (input instanceof Tensor) {
      if (this.inputDetails.length !== 1) {
        throw new Error(
          `run() called with a single tensor, but signature expects ${this.inputDetails.length} inputs`
        );
      }
      return [input];
    }
    const inputArray = [];
    for (const inputDetails of this.inputDetails) {
      if (!(inputDetails.name in input)) {
        throw new Error(
          `run() called with input record that is missing input ${inputDetails.name} with index ${inputDetails.index}`
        );
      }
      inputArray.push(input[inputDetails.name]);
    }
    return inputArray;
  }
  outputsToRecord(output) {
    const outputRecord = {};
    for (let i = 0; i < this.outputDetails.length; i++) {
      outputRecord[this.outputDetails[i].name] = output[i];
    }
    return outputRecord;
  }
  /**
   * Ensures that all input tensors are on the correct accelerator. Copies any
   * tensors that are not on the correct accelerator.
   *
   * @param inputs The input tensors to be passed to the signature. They must
   *     be in the same order and quantity as the input details.
   * @return A promise that resolves to a list of input tensors that are on the
   *     correct accelerator, and a cleanup function that deletes any tensors
   *     that were copied.
   */
  async ensureInputsOnAccelerator(inputs) {
    const toDelete = [];
    const inputsOnAccelerator = [];
    const inputDetails = this.getInputDetails();
    if (inputs.length !== inputDetails.length) {
      throw new Error(`ensureInputsOnAccelerator() called with ${inputs.length} inputs, but signature expects ${inputDetails.length} inputs`);
    }
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const bufferType = input.getBufferType();
      const supportedBufferTypes = inputDetails[i].supportedBufferTypes;
      if (supportedBufferTypes.size === 0) {
        throw new Error(`Tensor ${inputDetails[i].name} with index ${inputDetails[i].index} has no supported buffer types.`);
      }
      if (supportedBufferTypes.has(bufferType)) {
        inputsOnAccelerator.push(input);
      } else {
        const newBufferType = supportedBufferTypes.values().next().value;
        const copy = await input.copyTo(newBufferType);
        toDelete.push(copy);
        inputsOnAccelerator.push(copy);
      }
    }
    return {
      inputsOnAccelerator,
      cleanup: () => {
        for (const tensor of toDelete) {
          tensor.delete();
        }
      }
    };
  }
  async runWithArray(input) {
    for (let i = 0; i < input.length; i++) {
      const inputTensor = input[i];
      const expectedRankedTensorType = this.liteRtModel.getInputTensorType(this.signatureIndex, i);
      const inputRequirements = this.liteRtCompiledModel.getInputBufferRequirements(
        this.signatureIndex,
        i
      );
      getGlobalLiteRt().liteRtWasm.checkTensorBufferCompatible(
        inputTensor.liteRtTensorBuffer,
        expectedRankedTensorType,
        inputRequirements
      );
      expectedRankedTensorType.delete();
      inputRequirements.delete();
    }
    const outputTensorBuffers = await this.liteRtCompiledModel.run(
      this.signatureIndex,
      input.map((tensor) => tensor.liteRtTensorBuffer)
    );
    return outputTensorBuffers.map(
      (tensorBuffer) => new Tensor(tensorBuffer, this.options.environment)
    );
  }
  get deleted() {
    return this.deletedInternal;
  }
  ensureNotDeleted() {
    if (this.deleted) {
      throw new Error(
        "CompiledModelSignatureRunner is deleted and cannot be used."
      );
    }
  }
  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtSimpleSignature.delete();
  }
};
function makeTensorDetails(name, index, tensorType, requirements) {
  const layout = tensorType.layout();
  const dimensions = emscriptenVectorToArray(layout.dimensions());
  layout.delete();
  const supportedBufferTypes = new Set(emscriptenVectorToArray(requirements.supportedTypes()).map(({ value }) => value));
  const details = {
    name,
    index,
    dtype: getDataType(tensorType.elementType().value).dtype,
    shape: new Int32Array(dimensions),
    supportedBufferTypes
  };
  tensorType.delete();
  requirements.delete();
  return details;
}

// src/compiled_model.ts
var CompiledModel = class {
  constructor(model, liteRtCompiledModel, options, onDelete) {
    this.model = model;
    this.liteRtCompiledModel = liteRtCompiledModel;
    this.options = options;
    this.onDelete = onDelete;
    const numSignatures = model.liteRtModel.getNumSignatures();
    const compiledModelSignatureRunners = {};
    for (let i = 0; i < numSignatures; i++) {
      const compiledModelSignatureRunner = new CompiledModelSignatureRunner(
        i,
        model.liteRtModel,
        liteRtCompiledModel,
        options
      );
      compiledModelSignatureRunners[compiledModelSignatureRunner.key] = compiledModelSignatureRunner;
    }
    this.compiledModelSignatureRunners = Object.freeze(compiledModelSignatureRunners);
    this.defaultSignature = Object.values(this.signatures)[0];
    this.key = this.defaultSignature.key;
  }
  defaultSignature;
  compiledModelSignatureRunners;
  key;
  deletedInternal = false;
  get signatures() {
    this.ensureNotDeleted();
    return this.compiledModelSignatureRunners;
  }
  getInputDetails() {
    this.ensureNotDeleted();
    return this.defaultSignature.getInputDetails();
  }
  getOutputDetails() {
    this.ensureNotDeleted();
    return this.defaultSignature.getOutputDetails();
  }
  async run(inputOrSignatureName, maybeInput) {
    this.ensureNotDeleted();
    const [signature, input] = this.parseRunInputs(inputOrSignatureName, maybeInput);
    return await signature.run(input);
  }
  parseRunInputs(inputOrSignatureName, maybeInput) {
    let signature;
    let input;
    if (typeof inputOrSignatureName === "string") {
      signature = this.signatures[inputOrSignatureName];
      if (!signature) {
        throw new Error(
          `No signature named ${inputOrSignatureName} found in model.`
        );
      }
      if (!maybeInput) {
        throw new Error(
          `No input provided for signature ${inputOrSignatureName}`
        );
      }
      input = maybeInput;
    } else {
      signature = this.defaultSignature;
      input = inputOrSignatureName;
    }
    return [signature, input];
  }
  get deleted() {
    return this.deletedInternal;
  }
  ensureNotDeleted() {
    if (this.deleted) {
      throw new Error("CompiledModel is deleted and cannot be used.");
    }
  }
  get isFullyAccelerated() {
    this.ensureNotDeleted();
    return this.liteRtCompiledModel.isFullyAccelerated();
  }
  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtCompiledModel.delete();
    this.model.delete();
    for (const signatureRunner of Object.values(
      this.compiledModelSignatureRunners
    )) {
      signatureRunner.delete();
    }
    this.onDelete();
  }
};

// src/load_utils.ts
async function urlToUint8Array(url) {
  const response = await fetch(url);
  return new Uint8Array(await response.arrayBuffer());
}
async function readableStreamDefaultReaderToUint8Array(reader) {
  let byteOffset = 0;
  let array = new Uint8Array(
    1024
    /* arbitrary starting size */
  );
  const MAX_ARRAY_SIZE = 2e9;
  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      if (array.byteLength < byteOffset + value.byteLength) {
        if (byteOffset + value.byteLength > MAX_ARRAY_SIZE) {
          throw new Error(`Model is too large (> ${MAX_ARRAY_SIZE} bytes).`);
        }
        const newArray = new Uint8Array(Math.min(
          MAX_ARRAY_SIZE,
          Math.max(array.byteLength, value.byteLength) * 2
        ));
        newArray.set(array);
        array = newArray;
      }
      array.set(value, byteOffset);
      byteOffset += value.byteLength;
    }
    if (done) {
      break;
    }
  }
  return array.slice(0, byteOffset);
}

// src/model.ts
var Model = class {
  constructor(liteRtModel, onDelete) {
    this.liteRtModel = liteRtModel;
    this.onDelete = onDelete;
  }
  delete() {
    this.liteRtModel.delete();
    this.onDelete();
  }
};

// src/model_types.ts
function fillCompileOptions(compileOptions = {}, environment, defaultThreadCount) {
  return {
    environment,
    accelerator: compileOptions.accelerator ?? (environment.webGpuDevice ? "webgpu" : "wasm"),
    cpuOptions: compileOptions.cpuOptions ?? { numThreads: defaultThreadCount },
    gpuOptions: compileOptions.gpuOptions ?? {},
    webNNOptions: compileOptions.webNNOptions ?? {}
  };
}

// src/wasm_feature_detect.ts
var WASM_RELAXED_SIMD_CHECK = new Uint8Array([
  0,
  97,
  115,
  109,
  1,
  0,
  0,
  0,
  1,
  5,
  1,
  96,
  0,
  1,
  123,
  3,
  2,
  1,
  0,
  10,
  15,
  1,
  13,
  0,
  65,
  1,
  253,
  15,
  65,
  2,
  253,
  15,
  253,
  128,
  2,
  11
]);
var WASM_THREADS_CHECK = new Uint8Array([
  0,
  97,
  115,
  109,
  1,
  0,
  0,
  0,
  1,
  4,
  1,
  96,
  0,
  0,
  3,
  2,
  1,
  0,
  5,
  4,
  1,
  3,
  1,
  1,
  10,
  11,
  1,
  9,
  0,
  65,
  0,
  254,
  16,
  2,
  0,
  26,
  11
]);
var WASM_FEATURE_VALUES = {
  "relaxedSimd": void 0,
  "threads": void 0,
  "jspi": void 0,
  "webnn": void 0
};
function isJspiSupported() {
  return "Suspending" in WebAssembly;
}
function isWebNnSupported() {
  return typeof navigator !== "undefined" && !!navigator.ml;
}
async function tryWasm(wasm) {
  try {
    await WebAssembly.instantiate(wasm);
    return { supported: true };
  } catch (e) {
    return { supported: false, error: e };
  }
}
var WASM_FEATURE_CHECKS = {
  "relaxedSimd": () => {
    if (WASM_FEATURE_VALUES.relaxedSimd === void 0) {
      WASM_FEATURE_VALUES.relaxedSimd = tryWasm(WASM_RELAXED_SIMD_CHECK);
    }
    return WASM_FEATURE_VALUES.relaxedSimd;
  },
  "threads": () => {
    if (WASM_FEATURE_VALUES.threads === void 0) {
      try {
        if (typeof MessageChannel !== "undefined") {
          new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
        }
        WASM_FEATURE_VALUES.threads = tryWasm(WASM_THREADS_CHECK);
      } catch (e) {
        WASM_FEATURE_VALUES.threads = Promise.resolve({ supported: false, error: e });
      }
    }
    return WASM_FEATURE_VALUES.threads;
  },
  "jspi": () => {
    if (WASM_FEATURE_VALUES.jspi === void 0) {
      const supported = isJspiSupported();
      WASM_FEATURE_VALUES.jspi = Promise.resolve({
        supported,
        error: supported ? void 0 : new Error("JSPI is not supported")
      });
    }
    return WASM_FEATURE_VALUES.jspi;
  },
  "webnn": () => {
    if (WASM_FEATURE_VALUES.webnn === void 0) {
      const supported = isWebNnSupported();
      WASM_FEATURE_VALUES.webnn = Promise.resolve({
        supported,
        error: supported ? void 0 : new Error("WebNN is not supported")
      });
    }
    return WASM_FEATURE_VALUES.webnn;
  }
};
async function supportsFeature(feature) {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    throw new Error(`Unknown feature: ${feature}`);
  }
  return (await check).supported;
}
async function throwIfFeatureNotSupported(feature) {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    throw new Error(`Unknown feature: ${feature}`);
  }
  const result = await check;
  if (!result.supported) {
    throw result.error;
  }
}

// src/litert_web.ts
function isWebGPUSupported() {
  return !!(typeof globalThis !== "undefined" && globalThis.navigator && globalThis.navigator.gpu);
}
function getDefaultEnvironment() {
  return getGlobalLiteRt().getDefaultEnvironment();
}
function loadAndCompile(model, compileOptions) {
  return getGlobalLiteRt().loadAndCompile(model, compileOptions);
}
function getWebGpuDevice() {
  return getGlobalLiteRt().getWebGpuDevice();
}
function setWebGpuDevice(device) {
  getGlobalLiteRt().setWebGpuDevice(device);
}
var LiteRt = class {
  liteRtWasm;
  defaultEnvironment;
  objectsToDelete = /* @__PURE__ */ new Set();
  constructor(wasmModule) {
    this.liteRtWasm = wasmModule;
    this.liteRtWasm.setupLogging();
  }
  setDefaultEnvironment(environment) {
    this.defaultEnvironment = environment;
  }
  getDefaultEnvironment() {
    if (!this.defaultEnvironment) {
      throw new Error("Default environment is not set.");
    }
    return this.defaultEnvironment;
  }
  setWebGpuDevice(device) {
    const oldEnvironment = this.getDefaultEnvironment();
    this.setDefaultEnvironment(new Environment({
      ...oldEnvironment.options,
      webGpuDevice: device
    }));
  }
  getWebGpuDevice() {
    return this.getDefaultEnvironment().webGpuDevice;
  }
  /**
   * Registers an object to be deleted when this LiteRt instance is deleted.
   * Internal use only.
   */
  _registerObjectForDeletion(object) {
    this.objectsToDelete.add(object);
  }
  /**
   * Unregisters an object from being deleted when this LiteRt instance is
   * deleted. Internal use only.
   */
  _unregisterObjectForDeletion(object) {
    this.objectsToDelete.delete(object);
  }
  /**
   * Loads and compiles a LiteRt model.
   *
   * @param model The model data. This can be a string (the model url), a URL
   *     object, a Uint8Array (the model bytes), or a
   *     ReadableStreamDefaultReader (for streaming model loading).
   * @param compileOptions The options for compiling the model. This includes
   *     the accelerator to use ('webgpu' or 'wasm') and the WebGPU device
   *     (for direct GPU model inputs / outputs).
   * @returns A promise that resolves to the CompiledModel.
   */
  async loadAndCompile(model, compileOptions = {}) {
    let modelData;
    if (typeof model === "string" || model instanceof URL) {
      modelData = await urlToUint8Array(model);
    } else if (model instanceof Uint8Array) {
      modelData = model;
    } else if (model instanceof ReadableStreamDefaultReader) {
      modelData = await readableStreamDefaultReaderToUint8Array(model);
    } else {
      throw new Error("Unsupported model type.");
    }
    const environment = compileOptions.environment ?? this.getDefaultEnvironment();
    const accelerator = compileOptions.accelerator ?? (environment.webGpuDevice ? "webgpu" : "wasm");
    const isWebGpu = accelerator === "webgpu";
    if (isWebGpu && !environment.webGpuDevice) {
      throw new Error(
        "WebGPU was requested but no WebGPU device is set in the environment."
      );
    }
    const filledCompileOptions = fillCompileOptions(
      compileOptions,
      environment,
      this.liteRtWasm.getThreadCount()
    );
    const ptr = this.liteRtWasm._malloc(modelData.byteLength);
    this.liteRtWasm.HEAPU8.set(modelData, ptr);
    const wasmModel = this.liteRtWasm.loadModel(
      filledCompileOptions.environment.liteRtEnvironment,
      ptr,
      modelData.byteLength
    );
    const wasmCompiledModel = await this.liteRtWasm.compileModel(
      filledCompileOptions.environment.liteRtEnvironment,
      wasmModel,
      filledCompileOptions
    );
    const loadedModel = new Model(wasmModel, () => {
      this.liteRtWasm._free(ptr);
    });
    const compiledModel = new CompiledModel(
      loadedModel,
      wasmCompiledModel,
      filledCompileOptions,
      () => {
        this.objectsToDelete.delete(compiledModel);
      }
    );
    this.objectsToDelete.add(compiledModel);
    const isWebNn = accelerator === "webnn";
    const acceleratorRequested = isWebGpu || isWebNn;
    if (acceleratorRequested && !compiledModel.isFullyAccelerated) {
      if (isJspiSupported()) {
        console.warn(
          `%c[LiteRT]%c Model not fully compiled for ${accelerator}. Partially delegating to WASM execution.`,
          "background: #FFA000; color: black; font-weight: bold; padding: 2px 5px; border-radius: 3px;",
          "font-weight: bold;"
        );
      } else {
        console.warn(
          `%c[LiteRT]%c Model not fully compiled for ${accelerator} on non-JSPI browser. Falling back to WASM execution.`,
          "background: #D32F2F; color: white; font-weight: bold; padding: 2px 5px; border-radius: 3px;",
          "color: #D32F2F; font-weight: bold;"
        );
        compiledModel.delete();
        const fallbackCompileOptions = {
          ...compileOptions,
          accelerator: "wasm"
        };
        return this.loadAndCompile(modelData, fallbackCompileOptions);
      }
    }
    return compiledModel;
  }
  delete() {
    for (const object of this.objectsToDelete) {
      object.delete();
    }
  }
};

// src/load.ts
import { createWasmLib } from "./wasm-utils.js";

// src/url_path_utils.ts
function pathToString(path) {
  return path;
}
function appendPathSegment(path, segment) {
  if (!path) return segment;
  if (!segment) return path;
  const pathWithSlash = path.endsWith("/") ? path : path + "/";
  const segmentWithoutSlash = segment.startsWith("/") ? segment.substring(1) : segment;
  return pathWithSlash + segmentWithoutSlash;
}

// src/load.ts
var WASM_JS_FILE_NAME = "litert_wasm_internal.js";
var WASM_JS_COMPAT_FILE_NAME = "litert_wasm_compat_internal.js";
var WASM_JS_THREADED_FILE_NAME = "litert_wasm_threaded_internal.js";
var WASM_JS_JSPI_FILE_NAME = "litert_wasm_jspi_internal.js";
async function load(path, options) {
  const pathString = pathToString(path);
  const isFullFilePath = pathString.endsWith(".wasm") || pathString.endsWith(".js");
  const relaxedSimd = await supportsFeature("relaxedSimd");
  if (options?.threads) {
    if (options?.jspi) {
      throw new Error(
        "The `threads` and `jspi` options are mutually exclusive."
      );
    }
    if (isFullFilePath) {
      console.warn(
        `The \`threads\` option was specified, but the wasm path ${pathString} is a full file path. Whether threads are available or not will depend on the loaded file. To allow LiteRT.js to load the threaded wasm file, use a directory path instead of a full file path.`
      );
    }
    if (!relaxedSimd) {
      throw new Error(
        "Threads are only supported with relaxed SIMD, and the current browser does not support relaxed SIMD."
      );
    }
    await throwIfFeatureNotSupported("threads");
  }
  if (options?.jspi) {
    if (isFullFilePath) {
      console.warn(
        `The \`jspi\` option was specified, but the wasm path ${pathString} is a full file path. Whether JSPI is available or not will depend on the loaded file. To allow LiteRT.js to load the JSPI wasm file, use a directory path instead of a full file path.`
      );
    }
    await throwIfFeatureNotSupported("jspi");
  }
  let fileName = WASM_JS_COMPAT_FILE_NAME;
  if (relaxedSimd) {
    if (options?.threads) {
      fileName = WASM_JS_THREADED_FILE_NAME;
    } else if (options?.jspi) {
      fileName = WASM_JS_JSPI_FILE_NAME;
    } else {
      fileName = WASM_JS_FILE_NAME;
    }
  }
  let jsFilePath = path;
  if (pathString.endsWith(".wasm")) {
    throw new Error(
      "Please load the `.js` file corresponding to the `.wasm` file, or load the directory containing it."
    );
  } else if (!pathString.endsWith(".js")) {
    jsFilePath = appendPathSegment(path, fileName);
  }
  return createWasmLib(LiteRt, jsFilePath);
}

// src/load_litert.ts
function loadLiteRt(path, options) {
  if (hasGlobalLiteRtPromise()) {
    throw new Error("LiteRT is already loading / loaded.");
  }
  setGlobalLiteRtPromise(load(path, options).then(async (liteRt) => {
    setGlobalLiteRt(liteRt);
    liteRt.setDefaultEnvironment(
      await Environment.create()
    );
    return liteRt;
  }).catch((error) => {
    setGlobalLiteRtPromise(void 0);
    throw error;
  }));
  return getGlobalLiteRtPromise();
}
function unloadLiteRt() {
  if (hasGlobalLiteRtPromise() && !hasGlobalLiteRt()) {
    throw new Error(
      "LiteRT is loading and can not be unloaded or canceled until it is finished loading."
    );
  }
  if (hasGlobalLiteRt()) {
    getGlobalLiteRt().delete();
    setGlobalLiteRt(void 0);
  }
  setGlobalLiteRtPromise(void 0);
}

// src/streamed_loading.ts
var compilationLock = Promise.resolve();
async function loadModelAndWeights(modelData, weightsStream, compileOptions = {}) {
  const liteRt = getGlobalLiteRt();
  const wasm = liteRt.liteRtWasm;
  const env = compileOptions.environment ?? liteRt.getDefaultEnvironment();
  if (!env.webGpuDevice) {
    throw new Error(
      "WebGPU device is required for streamed loading in this implementation."
    );
  }
  const myTurn = compilationLock;
  let resolveLock;
  compilationLock = new Promise((resolve) => {
    resolveLock = resolve;
  });
  await myTurn;
  try {
    wasm.registerStreamWeightsCallback(async (tflIds, wgpuBufferIds, offsets, lengths) => {
      const requests = [];
      if (tflIds.length !== wgpuBufferIds.length) {
        throw new Error(
          `Stream weights callback received arrays of different lengths: tflIds=${tflIds.length}, wgpuBufferIds=${wgpuBufferIds.length}, `
        );
      }
      for (let i = 0; i < tflIds.length; i++) {
        requests.push({
          id: tflIds[i],
          wgpuBufferId: wgpuBufferIds[i],
          offset: offsets[i],
          length: lengths[i]
        });
      }
      requests.sort((a, b) => a.offset - b.offset);
      console.log(`[StreamWeights] Starting loading session for ${requests.length} tensors:`, JSON.stringify(requests));
      const reader = weightsStream.getReader();
      let streamOffset = 0;
      let buffer = new Uint8Array(0);
      let reqIndex = 0;
      try {
        while (reqIndex < requests.length) {
          const req = requests[reqIndex];
          const relStart = req.offset - streamOffset;
          const relEnd = relStart + req.length;
          if (relEnd <= buffer.length) {
            if (relStart < 0) {
              throw new Error(
                `Stream logic error: weight starts before current buffer (req.offset=${req.offset}, streamOffset=${streamOffset}).`
              );
            }
            let weightData = buffer.subarray(relStart, relEnd);
            if (weightData.byteLength % 4 !== 0) {
              const paddedSize = weightData.byteLength + 3 & ~3;
              const paddedData = new Uint8Array(paddedSize);
              paddedData.set(weightData);
              weightData = paddedData;
            }
            const gpuBuffer = wasm.WebGPU.getJsObject(req.wgpuBufferId);
            if (!gpuBuffer) {
              throw new Error(
                `Failed to find GPUBuffer for ID: ${req.wgpuBufferId}`
              );
            }
            console.log(`[StreamWeights] Writing ${weightData.byteLength} bytes to GPUBuffer ${req.wgpuBufferId} (tflId=${req.id}, first 5 bytes=${Array.from(weightData.slice(0, 5))})`);
            env.webGpuDevice.queue.writeBuffer(gpuBuffer, 0, weightData);
            reqIndex++;
          } else {
            const { done, value } = await reader.read();
            if (done) {
              throw new Error(
                `Stream ended before all weights were loaded.`
              );
            }
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;
          }
          if (reqIndex < requests.length) {
            const nextStartRel = requests[reqIndex].offset - streamOffset;
            const bytesToDiscard = Math.min(nextStartRel, buffer.length);
            if (bytesToDiscard > 1024 * 1024 || bytesToDiscard === buffer.length) {
              buffer = buffer.slice(bytesToDiscard);
              streamOffset += bytesToDiscard;
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    });
    try {
      const ptr = wasm._malloc(modelData.byteLength);
      wasm.HEAPU8.set(modelData, ptr);
      const wasmModel = wasm.loadModel(env.liteRtEnvironment, ptr, modelData.byteLength);
      const loadedModel = new Model(wasmModel, () => {
        wasm._free(ptr);
      });
      const fullOptions = fillCompileOptions(
        compileOptions,
        env,
        wasm.getThreadCount()
      );
      const wasmCompiledModel = await wasm.compileModel(
        env.liteRtEnvironment,
        wasmModel,
        fullOptions
      );
      const compiledModel = new CompiledModel(
        loadedModel,
        wasmCompiledModel,
        fullOptions,
        () => {
          liteRt._unregisterObjectForDeletion(compiledModel);
        }
      );
      liteRt._registerObjectForDeletion(compiledModel);
      return compiledModel;
    } finally {
      wasm.registerStreamWeightsCallback(void 0);
    }
  } finally {
    resolveLock();
  }
}

// src/cpu_copy_functions.ts
async function copyHostMemoryToHostMemory(cpuTensor, options = {}) {
  const environment = options.environment ?? cpuTensor.environment;
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  const srcTensorBuffer = cpuTensor.liteRtTensorBuffer;
  const bufferType = srcTensorBuffer.bufferType();
  if (bufferType.value !== TensorBufferType.HOST_MEMORY) {
    throw new Error(
      "Source tensor is not in host memory. Cannot copy to host memory."
    );
  }
  const srcTensorMemoryPtr = srcTensorBuffer.lock(
    liteRtWasm.LiteRtTensorBufferLockMode.READ
  );
  let destTensorBuffer;
  try {
    destTensorBuffer = liteRtWasm.LiteRtTensorBuffer.createManaged(
      environment.liteRtEnvironment,
      liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY,
      srcTensorBuffer.tensorType(),
      srcTensorBuffer.size()
    );
    const destMemoryPointer = destTensorBuffer.lock(
      liteRtWasm.LiteRtTensorBufferLockMode.WRITE
    );
    try {
      const srcTensorMemoryView = new Uint8Array(
        liteRtWasm.HEAPU8.buffer,
        srcTensorMemoryPtr,
        srcTensorBuffer.size()
      );
      liteRtWasm.HEAPU8.set(srcTensorMemoryView, destMemoryPointer);
    } finally {
      destTensorBuffer.unlock();
    }
  } finally {
    srcTensorBuffer.unlock();
  }
  if (!destTensorBuffer) {
    throw new Error("Failed to create destination tensor buffer.");
  }
  return new Tensor(destTensorBuffer, environment);
}

// src/gpu_copy_functions.ts
async function cpuTensorToGpuTensor(cpuTensor, options = {}) {
  const environment = options.environment ?? cpuTensor.environment;
  const device = environment.webGpuDevice;
  if (!device) {
    throw new Error(
      "No WebGPU device is available. Did you forget to pass a destination environment that has a WebGPU device?"
    );
  }
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  const byteLength = cpuTensor.liteRtTensorBuffer.size();
  const paddedByteLength = byteLength + 3 & ~3;
  const stagingBuffer = device.createBuffer({
    size: paddedByteLength,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true
  });
  const mappedBuffer = await stagingBuffer.getMappedRange();
  const mappedArray = new Uint8Array(mappedBuffer);
  const cpuMemoryPtr = cpuTensor.liteRtTensorBuffer.lock(
    liteRtWasm.LiteRtTensorBufferLockMode.READ
  );
  try {
    const cpuMemoryView = new Uint8Array(
      liteRtWasm.HEAPU8.buffer,
      cpuMemoryPtr,
      cpuTensor.liteRtTensorBuffer.size()
    );
    mappedArray.set(cpuMemoryView);
  } finally {
    cpuTensor.liteRtTensorBuffer.unlock();
  }
  stagingBuffer.unmap();
  const buffer = device.createBuffer({
    size: paddedByteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    stagingBuffer,
    0,
    buffer,
    0,
    paddedByteLength
  );
  device.queue.submit([commandEncoder.finish()]);
  stagingBuffer.destroy();
  return new Tensor(
    buffer,
    cpuTensor.type.layout.dimensions,
    cpuTensor.type.dtype,
    environment,
    () => {
      buffer.destroy();
    }
  );
}
async function gpuTensorToCpuTensor(gpuTensor, options = {}) {
  const environment = options.environment ?? gpuTensor.environment;
  const device = gpuTensor.environment.webGpuDevice;
  if (!device) {
    throw new Error(
      "No WebGPU device is available. Does the source tensor have a WebGPU device?"
    );
  }
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  const tensorBuffer = gpuTensor.liteRtTensorBuffer;
  const bufferType = tensorBuffer.bufferType();
  if (bufferType !== liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED) {
    throw new Error(`Cannot convert a tensor with a non-WebGPU buffer type ${bufferType} to a CPU tensor.`);
  }
  const gpuBuffer = liteRtWasm.WebGPU.getJsObject(
    tensorBuffer.getWebGpuBuffer()
  );
  const byteOffset = tensorBuffer.offset();
  const tensorType = tensorBuffer.tensorType();
  const layout = tensorType.layout();
  const numElements = layout.numElements();
  const arrayConstructor = getDataType(tensorType.elementType().value).typedArrayConstructor;
  layout.delete();
  tensorType.delete();
  let mappableBuffer = gpuBuffer;
  let cleanupBuffer = () => {
  };
  if (!(gpuBuffer.usage & GPUBufferUsage.MAP_READ)) {
    mappableBuffer = device.createBuffer({
      size: gpuBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    cleanupBuffer = () => {
      mappableBuffer.destroy();
    };
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      gpuBuffer,
      0,
      mappableBuffer,
      0,
      gpuBuffer.size
    );
    device.queue.submit([commandEncoder.finish()]);
  }
  await mappableBuffer.mapAsync(GPUMapMode.READ);
  const mappedBuffer = mappableBuffer.getMappedRange();
  const mappedArray = new arrayConstructor(mappedBuffer, byteOffset, numElements);
  const cpuTensor = new Tensor(mappedArray, gpuTensor.type.layout.dimensions, environment);
  mappableBuffer.unmap();
  cleanupBuffer();
  return cpuTensor;
}

// src/tensor_copy_functions.ts
function makeMoveTo(copyTo) {
  return async (tensor, options) => {
    const result = await copyTo(tensor, options);
    tensor.delete();
    return result;
  };
}
function registerCopyFunctions() {
  Tensor.copyFunctions.set(TensorBufferType.HOST_MEMORY, /* @__PURE__ */ new Map([
    [
      TensorBufferType.HOST_MEMORY,
      {
        copyTo: copyHostMemoryToHostMemory,
        // There might be a more efficient way to move
        // from CPU to CPU.
        moveTo: makeMoveTo(copyHostMemoryToHostMemory)
      }
    ],
    [
      TensorBufferType.WEB_GPU_BUFFER_PACKED,
      {
        copyTo: cpuTensorToGpuTensor,
        moveTo: makeMoveTo(cpuTensorToGpuTensor)
      }
    ]
  ]));
  Tensor.copyFunctions.set(TensorBufferType.WEB_GPU_BUFFER_PACKED, /* @__PURE__ */ new Map([
    [
      TensorBufferType.HOST_MEMORY,
      {
        copyTo: gpuTensorToCpuTensor,
        moveTo: makeMoveTo(gpuTensorToCpuTensor)
      }
    ]
  ]));
}

// src/index.ts
registerCopyFunctions();
export {
  CompiledModel,
  Environment,
  LiteRt,
  LiteRtNotLoadedError,
  Tensor,
  TensorBufferType,
  getDefaultEnvironment,
  getGlobalLiteRt,
  getGlobalLiteRtPromise,
  getWebGpuDevice,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  loadModelAndWeights,
  setWebGpuDevice,
  supportsFeature,
  unloadLiteRt
};
//# sourceMappingURL=index.js.map