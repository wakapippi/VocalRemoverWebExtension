// Safariでbench.htmlを実行してWebGPU対応状況と性能を検証する。
// 使い方: node run_bench_safari.mjs
// 事前に一度 `sudo safaridriver --enable` が必要 (またはSafariの開発メニューで
// 「リモートオートメーションを許可」をオン)。
import http from "http";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const DRIVER_PORT = 4444 + Math.floor(Math.random() * 1000);

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
console.log(`serving at http://127.0.0.1:${port}`);

const driver = spawn("safaridriver", ["-p", String(DRIVER_PORT)], { stdio: "inherit" });
await new Promise((s) => setTimeout(s, 1500));

const base = `http://127.0.0.1:${DRIVER_PORT}`;

async function wd(method, pathName, body) {
    const res = await fetch(base + pathName, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(`WebDriver ${method} ${pathName}: ${JSON.stringify(json).slice(0, 500)}`);
    return json.value;
}

let sessionId = null;
async function cleanup(code) {
    try { if (sessionId) await wd("DELETE", `/session/${sessionId}`); } catch (e) { }
    driver.kill();
    server.close();
    process.exit(code);
}

try {
    const session = await wd("POST", "/session", {
        capabilities: { alwaysMatch: { browserName: "safari" } },
    });
    sessionId = session.sessionId;
    console.log("safari session:", sessionId);

    await wd("POST", `/session/${sessionId}/url`, { url: `http://127.0.0.1:${port}/bench/bench.html` });

    // WebGPUの基本チェック
    const gpuInfo = await wd("POST", `/session/${sessionId}/execute/async`, {
        script: `const cb = arguments[arguments.length - 1];
            (async () => {
                if (!navigator.gpu) return { available: false, reason: "navigator.gpu undefined" };
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (!adapter) return { available: false, reason: "requestAdapter returned null" };
                    return { available: true };
                } catch (e) { return { available: false, reason: String(e) }; }
            })().then(cb);`,
        args: [],
    });
    console.log("WebGPU:", JSON.stringify(gpuInfo));

    if (!gpuInfo.available) await cleanup(1);

    // ベンチ完了までポーリング (ページ内ログも表示)
    let lastLogLen = 0;
    const deadline = Date.now() + 600000;
    let result = null;
    while (Date.now() < deadline) {
        await new Promise((s) => setTimeout(s, 3000));
        const state = await wd("POST", `/session/${sessionId}/execute/sync`, {
            script: `return { done: window.__benchDone === true, log: document.getElementById("log").textContent };`,
            args: [],
        });
        if (state.log.length > lastLogLen) {
            process.stdout.write(state.log.slice(lastLogLen));
            lastLogLen = state.log.length;
        }
        if (state.done) {
            result = await wd("POST", `/session/${sessionId}/execute/sync`, {
                script: "return window.__benchResult;",
                args: [],
            });
            break;
        }
    }

    console.log("\n===== SAFARI BENCHMARK RESULT =====");
    console.log(JSON.stringify(result, null, 2));
    await cleanup(result && result.ok ? 0 : 1);
} catch (e) {
    console.error(String(e.message || e));
    if (String(e).includes("Could not create a session") || String(e).includes("not allowed")) {
        console.error("\nリモートオートメーションが無効のようです。以下のいずれかを実行してください:");
        console.error("  1. ターミナルで: sudo safaridriver --enable");
        console.error("  2. Safariの「開発」メニュー → 「リモートオートメーションを許可」をオン");
    }
    await cleanup(1);
}
