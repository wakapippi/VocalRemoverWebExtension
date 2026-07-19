// ヘッドレスChromeでbench.htmlを実行して結果を表示する。
// 使い方: node run_bench.mjs
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".bin": "application/octet-stream",
    ".tflite": "application/octet-stream",
};

// リポジトリルートを配信する簡易静的サーバ (wasmスレッド用にCOOP/COEP付き)
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

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
console.log(`serving ${repoRoot} at http://127.0.0.1:${port}`);

const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: [
        "--enable-unsafe-webgpu",
        "--enable-features=WebGPU",
        "--use-angle=metal",
        "--no-sandbox",
    ],
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.text()));
page.on("pageerror", (err) => console.log("[pageerror]", err.message));

await page.goto(`http://127.0.0.1:${port}/bench/bench.html`, { waitUntil: "load" });
await page.waitForFunction("window.__benchDone === true", { timeout: 600000 });
const result = await page.evaluate("window.__benchResult");

console.log("\n===== BENCHMARK RESULT =====");
console.log(JSON.stringify(result, null, 2));

await browser.close();
server.close();
process.exit(result && result.ok ? 0 : 1);
