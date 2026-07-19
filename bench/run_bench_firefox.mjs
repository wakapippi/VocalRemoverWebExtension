// Firefoxでbench.htmlを実行してWebGPU対応状況と性能を検証する。
// 使い方: node run_bench_firefox.mjs [--headed]
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const FIREFOX = "/Applications/Firefox.app/Contents/MacOS/firefox";
const headed = process.argv.includes("--headed");

const MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".bin": "application/octet-stream",
    ".tflite": "application/octet-stream",
};

const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const filePath = path.join(repoRoot, urlPath);
    if (!filePath.startsWith(repoRoot) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404);
        res.end("not found");
        return;
    }
    res.writeHead(200, {
        "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
    });
    fs.createReadStream(filePath).pipe(res);
});
await new Promise((s) => server.listen(0, "127.0.0.1", s));
const port = server.address().port;
console.log(`serving at http://127.0.0.1:${port} (firefox ${headed ? "headed" : "headless"})`);

const defaultPrefs = process.argv.includes("--default-prefs");
const browser = await puppeteer.launch({
    browser: "firefox",
    executablePath: FIREFOX,
    headless: !headed,
    extraPrefsFirefox: defaultPrefs ? {} : {
        "dom.webgpu.enabled": true,
        "dom.webgpu.workers.enabled": true,
        "gfx.webgpu.ignore-blocklist": true,
    },
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.text()));
page.on("pageerror", (err) => console.log("[pageerror]", err.message));

// まずWebGPUの基本チェック
await page.goto(`http://127.0.0.1:${port}/bench/bench.html`, { waitUntil: "load" });
const gpuInfo = await page.evaluate(async () => {
    if (!navigator.gpu) return { available: false, reason: "navigator.gpu undefined" };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { available: false, reason: "requestAdapter returned null" };
        return { available: true, features: [...adapter.features].slice(0, 10) };
    } catch (e) {
        return { available: false, reason: String(e) };
    }
});
console.log("WebGPU:", JSON.stringify(gpuInfo));

let result = null;
if (gpuInfo.available) {
    await page.waitForFunction("window.__benchDone === true", { timeout: 600000 });
    result = await page.evaluate("window.__benchResult");
    console.log("\n===== FIREFOX BENCHMARK RESULT =====");
    console.log(JSON.stringify(result, null, 2));
}

await browser.close();
server.close();
process.exit(gpuInfo.available && result && result.ok ? 0 : 1);
