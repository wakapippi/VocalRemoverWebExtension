// FirefoxでE2Eテスト (e2e_test.html) を実行する。
// Firefoxの拡張機能ページCSP ('self' + 'wasm-unsafe-eval'、unsafe-evalなし) を
// /sandbox.html に適用して、CSP違反がないことも同時に検証する。
// 使い方: node run_e2e_firefox.mjs
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const FIREFOX = "/Applications/Firefox.app/Contents/MacOS/firefox";

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
    const headers = { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" };
    if (urlPath == "/sandbox.html") {
        // Firefox拡張機能ページ相当のCSPを適用 (manifest.firefox.jsonと同じ)
        headers["Content-Security-Policy"] = "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'";
    }
    res.writeHead(200, headers);
    fs.createReadStream(filePath).pipe(res);
});
await new Promise((s) => server.listen(0, "127.0.0.1", s));
const port = server.address().port;
console.log(`serving at http://127.0.0.1:${port} (sandbox.htmlに拡張CSP適用)`);

const browser = await puppeteer.launch({
    browser: "firefox",
    executablePath: FIREFOX,
    headless: false, // FirefoxのヘッドレスはWebGPUアダプタが取れない
    extraPrefsFirefox: { "dom.webgpu.enabled": true },
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.text()));
page.on("pageerror", (err) => console.log("[pageerror]", err.message));

await page.goto(`http://127.0.0.1:${port}/bench/e2e_test.html`, { waitUntil: "load" });
await page.waitForFunction("window.__e2eDone === true", { timeout: 300000 });
const result = await page.evaluate("window.__e2eResult");

console.log("\n===== FIREFOX E2E RESULT =====");
console.log(JSON.stringify(result, null, 2));

await browser.close();
server.close();
process.exit(result && result.ok ? 0 : 1);
