const { app, BrowserWindow, dialog, Menu } = require("electron");
const path = require("path");
const { spawn, execFileSync, spawnSync } = require("child_process");

const PORT = parseInt(process.env.FAMILYPHOTOS_PORT || "8765", 10);
/** Repo root: .../familyphotos.ai */
const REPO_ROOT = path.join(__dirname, "..", "..");

let backendProcess = null;
let mainWindow = null;

/**
 * Kill the backend and every child (uv → python). Required on Windows: with shell:true,
 * child.kill() often stops only cmd.exe and leaves Python holding CUDA VRAM.
 */
function stopBackendProcess() {
  if (!backendProcess) return;
  const pid = backendProcess.pid;
  backendProcess = null;

  if (process.platform === "win32") {
    try {
      execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
    } catch {
      /* already exited */
    }
    return;
  }

  const script = `require("tree-kill")(${pid}, "SIGTERM", () => process.exit(0)); setTimeout(() => process.exit(0), 8000);`;
  spawnSync(process.execPath, ["-e", script], {
    stdio: "ignore",
    cwd: __dirname,
    env: process.env,
  });
}

function startBackend() {
  return new Promise((resolve, reject) => {
    const child = spawn("uv", ["run", "python", "desktop/backend/main.py"], {
      cwd: REPO_ROOT,
      env: { ...process.env, FAMILYPHOTOS_PORT: String(PORT) },
      // Avoid cmd.exe wrapper so our PID is uv's and taskkill /T reliably tears down Python.
      shell: false,
      stdio: "inherit",
      windowsHide: true,
    });
    backendProcess = child;
    child.on("error", reject);
    child.on("spawn", () => resolve());
  });
}

/** Long timeout: first launch downloads Hugging Face weights before the API binds. */
async function waitForHealth(maxMs = 180000) {
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

function createMainWindowShell() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 760,
    backgroundColor: "#e9ecef",
    show: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
}

function showLoadingScreen() {
  createMainWindowShell();
  mainWindow.loadFile(path.join(__dirname, "loading.html"));
}

function loadMainApp() {
  return mainWindow.loadFile(path.join(__dirname, "index.html"), {
    query: { port: String(PORT) },
  });
}

/** macOS: reopen main UI when all windows were closed but the app keeps running. */
function recreateMainAppWindow() {
  createMainWindowShell();
  return loadMainApp();
}

app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  showLoadingScreen();
  try {
    await startBackend();
    await waitForHealth();
    await loadMainApp();
  } catch (e) {
    console.error(e);
    const detail = e && e.message ? e.message : String(e);
    if (mainWindow && !mainWindow.isDestroyed()) {
      await dialog.showMessageBox(mainWindow, {
        type: "error",
        title: "Could not start familyphotos.ai",
        message: "The app could not finish loading.",
        detail,
      });
    } else {
      dialog.showErrorBox("Could not start familyphotos.ai", detail);
    }
    app.quit();
    return;
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) recreateMainAppWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  stopBackendProcess();
});
