// Chrome Web Store用スクリーンショット (640x400 / 1280x800) を生成する。
// 実物のポップアップ (index.html) をchrome APIスタブ付きで「動作中」状態にして撮影する。
// 使い方: cd store && node make_screenshot.mjs   (bench/のnode_modulesを使用)
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

// 実物のindex.htmlにchrome APIスタブを注入したプレビュー版を生成
const stub = `<script>
window.chrome = {
    storage: { local: {
        get: (defaults, cb) => cb({ vocalRatio: 30, instRatio: 100, denoise: true }),
        set: () => {},
    }},
    runtime: {
        sendMessage: (msg, cb) => { if (msg.type == "getStatus" && cb) cb({ running: true }); },
        lastError: undefined,
    },
    tabs: { query: (q, cb) => cb([{ id: 1 }]), sendMessage: (id, msg, cb) => cb("ok") },
};
</script>
<script src="../index.js"></script>`;

const popupHtml = fs.readFileSync(path.join(repoRoot, "index.html"), "utf8")
    .replace('<script src="index.js"></script>', stub);
fs.writeFileSync(path.join(__dirname, "_popup_preview.html"), popupHtml);

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const filePath = path.join(repoRoot, urlPath);
    if (!filePath.startsWith(repoRoot) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404);
        res.end();
        return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    fs.createReadStream(filePath).pipe(res);
});
await new Promise((s) => server.listen(0, "127.0.0.1", s));
const port = server.address().port;

const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--no-sandbox", "--hide-scrollbars"] });

for (const [file, dsf] of [["screenshot_640x400.png", 1], ["screenshot_1280x800.png", 2]]) {
    const page = await browser.newPage();
    await page.setViewport({ width: 640, height: 400, deviceScaleFactor: dsf });
    await page.goto(`http://127.0.0.1:${port}/store/screenshot.html`, { waitUntil: "networkidle0" });
    await new Promise((s) => setTimeout(s, 600)); // フォント・アニメーション安定待ち
    await page.screenshot({ path: path.join(__dirname, file), clip: { x: 0, y: 0, width: 640, height: 400 } });
    console.log("saved:", file);
    await page.close();
}

await browser.close();
server.close();
