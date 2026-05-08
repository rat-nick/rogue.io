'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  startTraining:       (opts)   => ipcRenderer.invoke('start-training', opts),
  stopTraining:        ()       => ipcRenderer.invoke('stop-training'),
  isTrainingRunning:   ()       => ipcRenderer.invoke('is-training-running'),
  readConfig:          (p)      => ipcRenderer.invoke('read-config', p),
  writeConfig:         (args)   => ipcRenderer.invoke('write-config', args),
  readTrainingConfig:  (p)      => ipcRenderer.invoke('read-training-config', p),
  exportLog:           (args)   => ipcRenderer.invoke('export-log', args),

  onServerLog:         (cb) => ipcRenderer.on('server-log',     (_, d) => cb(d)),
  onServerStopped:     (cb) => ipcRenderer.on('server-stopped', (_, d) => cb(d)),
  onMenuStartTraining: (cb) => ipcRenderer.on('menu-start-training', () => cb()),
  onMenuStopTraining:  (cb) => ipcRenderer.on('menu-stop-training',  () => cb()),
  onMenuNextGen:       (cb) => ipcRenderer.on('menu-next-gen',       () => cb()),
});
