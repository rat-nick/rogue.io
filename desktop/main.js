'use strict';

const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron');
const path   = require('path');
const { spawn, execFile } = require('child_process');
const fs     = require('fs');

let mainWindow   = null;
let trainProcess = null;
let alwaysOnTop  = false;

// ---- Window ----------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width:  1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#111',
    title: 'rogue.io — Training',
    webPreferences: {
      preload:            path.join(__dirname, 'preload.js'),
      contextIsolation:   true,
      nodeIntegration:    false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  buildMenu();

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopTraining();
  });
}

function buildMenu() {
  const template = [
    {
      label: 'Training',
      submenu: [
        {
          label: 'Start Training Server',
          accelerator: 'CmdOrCtrl+R',
          click: () => mainWindow && mainWindow.webContents.send('menu-start-training'),
        },
        {
          label: 'Stop Training Server',
          accelerator: 'CmdOrCtrl+.',
          click: () => mainWindow && mainWindow.webContents.send('menu-stop-training'),
        },
        { type: 'separator' },
        {
          label: 'Next Generation',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow && mainWindow.webContents.send('menu-next-gen'),
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        {
          label: 'Always on Top',
          type: 'checkbox',
          checked: false,
          click: (item) => {
            alwaysOnTop = item.checked;
            if (mainWindow) mainWindow.setAlwaysOnTop(alwaysOnTop);
          },
        },
        { type: 'separator' },
        { role: 'toggleDevTools' },
        { role: 'reload' },
        { type: 'separator' },
        { role: 'togglefullscreen', accelerator: 'F11' },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Open Project Folder',
          click: () => shell.openPath(path.join(__dirname, '..')),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---- IPC: server process management ----------------------------------------

ipcMain.handle('start-training', async (_, opts) => {
  if (trainProcess) return { ok: false, error: 'Already running' };

  // opts: { pythonPath, projectPath }
  const python  = opts.pythonPath || 'python';
  const projDir = opts.projectPath || path.join(__dirname, '..');

  try {
    trainProcess = spawn(python, ['-m', 'server.train'], {
      cwd: projDir,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    trainProcess.stdout.on('data', (d) => {
      if (mainWindow) mainWindow.webContents.send('server-log', { level: 'info', text: d.toString() });
    });
    trainProcess.stderr.on('data', (d) => {
      if (mainWindow) mainWindow.webContents.send('server-log', { level: 'error', text: d.toString() });
    });
    trainProcess.on('exit', (code) => {
      trainProcess = null;
      if (mainWindow) mainWindow.webContents.send('server-stopped', { code });
    });
    trainProcess.on('error', (err) => {
      trainProcess = null;
      if (mainWindow) mainWindow.webContents.send('server-stopped', { code: -1, error: err.message });
    });

    return { ok: true, pid: trainProcess.pid };
  } catch (err) {
    trainProcess = null;
    return { ok: false, error: err.message };
  }
});

ipcMain.handle('stop-training', async () => {
  stopTraining();
  return { ok: true };
});

function stopTraining() {
  if (trainProcess) {
    try { trainProcess.kill(); } catch (_) {}
    trainProcess = null;
  }
}

ipcMain.handle('is-training-running', () => ({ running: trainProcess !== null }));

// ---- IPC: config read/write ------------------------------------------------

ipcMain.handle('read-config', async (_, projectPath) => {
  const cfgPath = path.join(projectPath, 'server', 'config.py');
  try {
    const text = fs.readFileSync(cfgPath, 'utf8');
    return { ok: true, text };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle('write-config', async (_, { projectPath, text, file }) => {
  const cfgPath = path.join(projectPath, file || 'server/config.py');
  try {
    fs.writeFileSync(cfgPath, text, 'utf8');
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle('read-training-config', async (_, projectPath) => {
  const cfgPath = path.join(projectPath, 'server', 'training.py');
  try {
    const text = fs.readFileSync(cfgPath, 'utf8');
    return { ok: true, text };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});
// ---- IPC: export log -------------------------------------------------------

ipcMain.handle('export-log', async (_, { projectPath, data }) => {
  const logsDir = path.join(projectPath, 'training_logs');
  try {
    fs.mkdirSync(logsDir, { recursive: true });
    const fname = `training_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    fs.writeFileSync(path.join(logsDir, fname), JSON.stringify(data, null, 2), 'utf8');
    return { ok: true, path: path.join(logsDir, fname) };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

// ---- App lifecycle ---------------------------------------------------------

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  stopTraining();
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
