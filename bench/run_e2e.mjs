// ヘッドレスChromeでe2e_test.html (sandbox.jsのパイプライン一式) を実行する。
// 使い方: node run_e2e.mjs
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

const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const filePath = path.join(repoRoot, urlPath);
    if (!filePath.startsWith(repoRoot) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404);
        res.end("not found");
        return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    fs.createReadStream(filePath).pipe(res);
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
console.log(`serving ${repoRoot} at http://127.0.0.1:${port}`);

const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "--no-sandbox"],
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.text()));
page.on("pageerror", (err) => console.log("[pageerror]", err.message));
// iframe (sandbox) のコンソールも拾う
browser.on("targetcreated", async (t) => {
    try {
        const p = await t.page();
        if (p) p.on("console", (msg) => console.log("[frame]", msg.text()));
    } catch (e) { }
});

await page.goto(`http://127.0.0.1:${port}/bench/e2e_test.html`, { waitUntil: "load" });
await page.waitForFunction("window.__e2eDone === true", { timeout: 300000 });
const result = await page.evaluate("window.__e2eResult");

console.log("\n===== E2E RESULT =====");
console.log(JSON.stringify(result, null, 2));

await browser.close();
server.close();
process.exit(result && result.ok ? 0 : 1);
