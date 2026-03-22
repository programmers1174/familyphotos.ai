const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

const PORT = parseInt(process.env.FAMILYPHOTOS_PORT || "8765", 10);
/** Repo root: .../familyphotos.ai */
const REPO_ROOT = path.join(__dirname, "..", "..");

let backendProcess = null;
let mainWindow = null;

function startBackend() {
  return new Promise((resolve, reject) => {
    const child = spawn("uv", ["run", "python", "desktop/backend/main.py"], {
      cwd: REPO_ROOT,
      env: { ...process.env, FAMILYPHOTOS_PORT: String(PORT) },
      shell: true,
      stdio: "inherit",
    });
    backendProcess = child;
    child.on("error", reject);
    child.on("spawn", () => resolve());
  });
}

async function waitForHealth(maxMs = 45000) {
  const url = `http://127.0.0.1:${PORT}/api/health`;
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`Backend did not become ready at ${url}`);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 760,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"), {
    query: { port: String(PORT) },
  });
}

app.whenReady().then(async () => {
  try {
    await startBackend();
    await waitForHealth();
  } catch (e) {
    console.error(e);
    app.quit();
    return;
  }
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
    backendProcess = null;
  }
});
