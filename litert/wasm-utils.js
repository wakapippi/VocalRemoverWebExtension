// src/run_script_helper.ts
async function runScript(scriptUrl) {
  if (typeof importScripts === "function") {
    importScripts(scriptUrl.toString());
  } else {
    const script = document.createElement("script");
    script.src = scriptUrl.toString();
    script.crossOrigin = "anonymous";
    return new Promise((resolve, revoke) => {
      script.addEventListener("load", () => {
        resolve();
      }, false);
      script.addEventListener("error", (e) => {
        revoke(e);
      }, false);
      document.body.appendChild(script);
    });
  }
}

// src/wasm_loader.ts
var createWasmLib = async (constructorFcn, wasmLoaderScript, assetLoaderScript, glCanvas, fileLocator) => {
  if (wasmLoaderScript) {
    await runScript(wasmLoaderScript);
  }
  if (!self.ModuleFactory) {
    throw new Error("ModuleFactory not set.");
  }
  if (assetLoaderScript) {
    await runScript(assetLoaderScript);
    if (!self.ModuleFactory) {
      throw new Error("ModuleFactory not set.");
    }
  }
  if (self.Module && fileLocator) {
    const moduleFileLocator = self.Module;
    moduleFileLocator.locateFile = fileLocator.locateFile;
    if (fileLocator.mainScriptUrlOrBlob) {
      moduleFileLocator.mainScriptUrlOrBlob = fileLocator.mainScriptUrlOrBlob;
    }
  }
  const module = await self.ModuleFactory(self.Module || fileLocator);
  self.ModuleFactory = self.Module = void 0;
  return new constructorFcn(module, glCanvas);
};
export {
  createWasmLib
};
//# sourceMappingURL=index.js.map