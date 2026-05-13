'use strict';
/**
 * app.js — renderer process controller for rogue.io Training Desktop App
 *
 * Features beyond the web client:
 *   - Start/stop training server process
 *   - Fitness & mass charts (Chart.js) with per-generation history
 *   - Config editor (training.py + config.py) with live parsing + save
 *   - Server log viewer
 *   - Export training history as JSON
 *   - Keyboard shortcuts
 *   - Auto-reconnect WebSocket
 *   - Always-on-top via menu
 */

(() => {
  // ── Constants ──────────────────────────────────────────────────────────────
  const MSG_INIT                = 0x10;
  const MSG_TICK                = 0x11;
  const MSG_TRAINING_STATS      = 0x24;
  const MSG_NEXT_GEN            = 0x25;
  const MSG_SET_EARLY_NEXT_GEN  = 0x26;
  const MSG_SET_TIME_SCALE      = 0x27;
  const MSG_SET_TRAINING_PARAMS = 0x28;
  const MSG_PPO_STATS           = 0x30;  // PPO training server only (port 8767)

  const ZOOM_LEVELS  = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0];
  const ZOOM_LABELS  = ['0.25×', '0.5×', '0.75×', '1×', '1.5×', '2.5×', '4×'];
  const GENERATION_TIME = 60.0; // fallback; overridden when cfg loaded

  // ── App State ──────────────────────────────────────────────────────────────
  let ws             = null;
  let wsPort         = 8766;
  let reconnectTimer = null;
  let wsConnected    = false;
  let trainMode      = 'neat';   // 'neat' | 'ppo'

  let followId       = null;
  let zoomIdx        = 3;

  let trainGen       = 0;
  let trainTimeLeft  = 0;
  let trainPop       = 0;
  let trainTopFit    = 0;
  let trainAvgFit    = 0;
  let trainBestMass  = 0;
  let trainAvgMass   = 0;
  let trainDeaths    = 0;
  let trainBots      = [];   // [[id, name, mass, cx, cy, cells, is_bot, fitness], ...]
  let sortMode       = 'mass'; // 'mass' | 'fitness'
  let trainFood      = 0;
  let genTimeSec     = GENERATION_TIME;
  let selectedSpecies = null;
  const playerSpeciesMap = new Map(); // pid -> species_id
  let chartRange = 50;           // 0 = all gens
  let currentSpeciesSnapshot = {}; // speciesId -> {count,topFit,totalFit,bestMass,avgFit}
  let spSortMode = 'count';      // 'count' | 'fitness'

  // PPO-specific stats
  let ppoRollout   = 0;
  let ppoEnvSteps  = 0;
  let ppoPgLoss    = 0;
  let ppoVfLoss    = 0;
  let ppoEntropy   = 0;
  let ppoKl        = 0;

  // Generation history for charts
  let history = {
    generations:      [],
    topFitness:       [],
    avgFitness:       [],
    bestMass:         [],
    avgMass:          [],
    deaths:           [],
    popSize:          [],
    speciesSnapshots: [], // per-gen: {speciesId -> {count,topFit,avgFit,bestMass}}
  };
  let allTimeBestFit   = 0;
  let allTimeBestMass  = 0;
  let totalDeathsEver  = 0;
  let lastRecordedGen  = -1;
  // Current-gen fitness distribution (bot fitnesses sampled per stats update)
  let currentGenFitnesses = [];

  // TPS (broadcast rate from MSG_TICK count; sim TPS derived from stats tick_num delta)
  let tpsTicks      = 0;
  let tpsStart      = performance.now();
  let tps           = 0;
  let simTps        = 0;
  let lastStatsTick = -1;
  let lastStatsTime = 0;

  // Persisted settings
  let projectPath   = '';
  let pythonPath    = 'python';
  let serverRunning = false;
  let earlyNextGen  = true;
  let timeScale     = 1.0;

  // ── Controls tab state ────────────────────────────────────────────────────
  const CTRL_DEFS = [
    { group: 'Evolution', items: [
      { key: 'survive_fraction', label: 'Survive Fraction',    min: 0.05, max: 0.9,   step: 0.05,  def: 0.35  },
      { key: 'diversity_rate',   label: 'Diversity Rate',      min: 0,    max: 0.5,   step: 0.01,  def: 0.10  },
      { key: 'tournament_k',     label: 'Tournament K',        min: 2,    max: 12,    step: 1,     def: 4     },
      { key: 'pop_size',         label: 'Population Size',     min: 10,   max: 500,   step: 10,    def: 150   },
    ]},
    { group: 'Generation', items: [
      { key: 'generation_time',  label: 'Gen Time (s)',        min: 30,   max: 600,   step: 10,    def: 120   },
      { key: 'early_threshold',  label: 'Early End Threshold', min: 0.1,  max: 0.95,  step: 0.05,  def: 0.40  },
    ]},
    { group: 'Fitness Weights', items: [
      { key: 'fw_food',          label: 'Food Eaten ×',        min: 0,    max: 20,    step: 0.5,   def: 3.0   },
      { key: 'fw_cells',         label: 'Cells Eaten ×',       min: 0,    max: 30,    step: 0.5,   def: 9.0   },
      { key: 'fw_survival',      label: 'Survival Bonus ×',    min: 0,    max: 20,    step: 0.5,   def: 3.0   },
      { key: 'fw_peak',          label: 'Peak Mass ×',         min: 0,    max: 5,     step: 0.1,   def: 0.4   },
      { key: 'fw_avg',           label: 'Avg Mass ×',          min: 0,    max: 5,     step: 0.1,   def: 0.3   },
      { key: 'fw_corner',        label: 'Corner Penalty ×',    min: 0,    max: 5000,  step: 50,    def: 1000  },
      { key: 'fw_death_exp',     label: 'Death Penalty Exp',   min: 0,    max: 10,    step: 0.25,  def: 3.0   },
    ]},
    { group: 'Hebbian Learning', items: [
      { key: 'hebbian_lr',       label: 'Learning Rate',       min: 0,    max: 0.05,  step: 0.001, def: 0.004 },
      { key: 'hebbian_decay',    label: 'Trace Decay',         min: 0.5,  max: 0.99,  step: 0.01,  def: 0.92  },
    ]},
    { group: 'NEAT Mutation', items: [
      { key: 'weight_mutate_power',  label: 'Weight Perturb Magnitude', min: 0,    max: 3,    step: 0.05,  def: 0.3  },
      { key: 'weight_mutate_rate',   label: 'Weight Mutate Rate',       min: 0,    max: 1,    step: 0.01,  def: 0.3  },
      { key: 'weight_replace_rate',  label: 'Weight Replace Rate',      min: 0,    max: 0.2,  step: 0.005, def: 0.02 },
      { key: 'conn_add_prob',        label: 'Add Connection Prob',      min: 0,    max: 0.5,  step: 0.01,  def: 0.10 },
      { key: 'conn_delete_prob',     label: 'Delete Connection Prob',   min: 0,    max: 0.2,  step: 0.005, def: 0.02 },
      { key: 'node_add_prob',        label: 'Add Node Prob',            min: 0,    max: 0.2,  step: 0.005, def: 0.03 },
      { key: 'node_delete_prob',     label: 'Delete Node Prob',         min: 0,    max: 0.1,  step: 0.002, def: 0.01 },
      { key: 'compat_threshold',     label: 'Species Compat Threshold', min: 0.5,  max: 10,   step: 0.25,  def: 2.0  },
    ]},
  ];
  let ctrlValues  = {};       // key -> current value (server-confirmed or default)
  let ctrlDirty   = new Set(); // keys with changes pending server confirmation
  let _ctrlBuilt  = false;
  let _ctrlTimer  = null;

  // ── DOM ────────────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const gameCanvas   = $('game');
  const minimapCvs   = $('minimap-canvas');
  const genBanner    = $('gen-banner');
  const followBadge  = $('follow-badge');
  const zoomLabel    = $('zoom-label');

  const sSgen    = $('s-gen');
  const sStime   = $('s-time');
  const sStimebar= $('s-timebar');
  const sSTopfit = $('s-topfit');
  const sSAvgfit = $('s-avgfit');
  const sSBmass  = $('s-bmass');
  const sSAmass  = $('s-amass');
  const sSPop    = $('s-pop');
  const sSDeaths = $('s-deaths');
  const sSFood   = $('s-food');

  // PPO stat elements
  const ppoSection   = $('ppo-section');
  const genSection   = $('gen-section');
  const rowTopFit    = $('row-topfit');
  const rowDeaths    = $('row-deaths');
  const ppoPanelRollout = $('ppo-rollout');
  const ppoPanelSteps   = $('ppo-env-steps');
  const ppoPanelPg      = $('ppo-pg-loss');
  const ppoPanelVf      = $('ppo-vf-loss');
  const ppoPanelEnt     = $('ppo-entropy');
  const ppoPanelKl      = $('ppo-kl');
  const modeNeatBtn  = $('mode-neat');
  const modePpoBtn   = $('mode-ppo');

  const fName    = $('follow-name');
  const fDetail  = $('follow-detail');
  const fMass    = $('f-mass');
  const fCells   = $('f-cells');
  const fRank    = $('f-rank');

  const botList       = $('bot-list');
  const botCount      = $('bot-count');
  const botSortToggle = $('bot-sort-toggle');

  const tpsDisplay  = $('tps-display');
  const statusDot   = $('status-dot');
  const statusText  = $('status-text');

  const btnStart        = $('btn-start');
  const btnStop         = $('btn-stop');
  const btnNextGen      = $('btn-nextgen');
  const btnEarlyNextGen = $('btn-early-nextgen');
  const speedBtns       = document.querySelectorAll('.speed-btn');
  const logOut    = $('log-out');

  // Chart summaries
  const scGen      = $('sc-gen');
  const scBestFit  = $('sc-bestfit');
  const scBestMass = $('sc-bestmass');
  const scTotalDead= $('sc-totaldead');

  // ── Init subsystems ────────────────────────────────────────────────────────
  Renderer.init(gameCanvas);
  Minimap.init(minimapCvs);
  SpectateViewport.init(gameCanvas);

  window.addEventListener('resize', () => { Renderer.resizeCanvas(); });
  Renderer.resizeCanvas();

  // ── Tabs ───────────────────────────────────────────────────────────────────
  let activeTab = 'viewer';
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `pane-${tab}`));
    if (tab === 'charts')   refreshCharts();
    if (tab === 'config')   loadConfigValues();
    if (tab === 'controls') _initControlsTab();
  }

  // ── Mode switching ─────────────────────────────────────────────────────────
  function switchTrainMode(mode) {
    if (mode === trainMode) return;
    trainMode = mode;
    wsPort = mode === 'ppo' ? 8767 : 8766;

    modeNeatBtn.classList.toggle('active', mode === 'neat');
    modePpoBtn.classList.toggle('active',  mode === 'ppo');

    const isPpo = mode === 'ppo';
    if (ppoSection)  ppoSection.style.display  = isPpo ? '' : 'none';
    if (genSection)  genSection.style.display  = isPpo ? 'none' : '';
    if (rowTopFit)   rowTopFit.style.display   = isPpo ? 'none' : '';
    if (rowDeaths)   rowDeaths.style.display   = isPpo ? 'none' : '';
    btnNextGen.style.display      = isPpo ? 'none' : '';
    btnEarlyNextGen.style.display = isPpo ? 'none' : '';
    genBanner.textContent = isPpo
      ? 'Rollout —  |  — steps'
      : 'Gen — | —s';

    // Reset live stats
    trainBots = [];
    _rebuildBotList();

    // Reconnect to the correct port if already connected
    if (wsConnected || ws) {
      connectWS();
    }
  }

  if (modeNeatBtn) modeNeatBtn.addEventListener('click', () => switchTrainMode('neat'));
  if (modePpoBtn)  modePpoBtn.addEventListener('click',  () => switchTrainMode('ppo'));

  // ── Zoom ───────────────────────────────────────────────────────────────────
  function applyZoom(idx) {
    zoomIdx = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, idx));
    SpectateViewport.setZoomMultiplier(ZOOM_LEVELS[zoomIdx]);
    zoomLabel.textContent = ZOOM_LABELS[zoomIdx];
  }
  $('zoom-in') .addEventListener('click', () => applyZoom(zoomIdx + 1));
  $('zoom-out').addEventListener('click', () => applyZoom(zoomIdx - 1));
  gameCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    applyZoom(e.deltaY < 0 ? zoomIdx + 1 : zoomIdx - 1);
  }, { passive: false });
  applyZoom(3);

  // ── Bot list sort toggle ───────────────────────────────────────────────────
  if (botSortToggle) {
    botSortToggle.addEventListener('click', () => {
      sortMode = sortMode === 'mass' ? 'fitness' : 'mass';
      botSortToggle.textContent = sortMode === 'fitness' ? 'Fitness ↓' : 'Mass ↓';
      botSortToggle.style.color = sortMode === 'fitness' ? '#fa4' : 'rgba(255,255,255,0.55)';
      botSortToggle.style.borderColor = sortMode === 'fitness' ? 'rgba(250,170,0,0.5)' : 'rgba(255,255,255,0.15)';
      _rebuildBotList();
    });
  }

  // ── Follow ─────────────────────────────────────────────────────────────────
  function setFollow(id) {
    followId = id || null;
    _updateFollowUI();
    _rebuildBotList();
  }

  function _updateFollowUI() {
    if (followId) {
      const p = trainBots.find(b => b[0] === followId);
      const name = p ? p[1] : `Bot #${followId}`;
      fName.textContent   = name;
      followBadge.textContent = `Following: ${name}`;
      fDetail.style.display   = '';
    } else {
      fName.textContent   = '— Overview —';
      followBadge.textContent = 'Overview';
      fDetail.style.display   = 'none';
    }
  }

  // ── Server control ─────────────────────────────────────────────────────────
  btnStart.addEventListener('click', startServer);
  btnStop .addEventListener('click', stopServer);
  btnNextGen.addEventListener('click', sendNextGen);
  btnEarlyNextGen.addEventListener('click', toggleEarlyNextGen);

  function _syncEarlyNextGenButton() {
    btnEarlyNextGen.textContent = earlyNextGen ? '⚡ Auto End: ON' : '⚡ Auto End: OFF';
    btnEarlyNextGen.className   = 'tb-btn ' + (earlyNextGen ? 'active-toggle' : 'inactive-toggle');
  }

  function toggleEarlyNextGen() {
    earlyNextGen = !earlyNextGen;
    _syncEarlyNextGenButton();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(msgpack.encode([MSG_SET_EARLY_NEXT_GEN, earlyNextGen]));
    }
  }

  function setTimeScale(scale) {
    timeScale = scale;
    genTimeSec = (parseFloat($('tc-gen-time').value) || GENERATION_TIME) / timeScale;
    speedBtns.forEach(btn => {
      const active = parseFloat(btn.dataset.scale) === timeScale;
      btn.classList.toggle('active-toggle', active);
      btn.classList.toggle('inactive-toggle', !active);
    });
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log(`[speed] sending time_scale=${timeScale}`);
      ws.send(msgpack.encode([MSG_SET_TIME_SCALE, timeScale]));
    } else {
      console.warn(`[speed] WebSocket not open (state=${ws ? ws.readyState : 'null'}), time_scale not sent`);
    }
  }

  speedBtns.forEach(btn => {
    btn.addEventListener('click', () => setTimeScale(parseFloat(btn.dataset.scale)));
  });

  async function startServer() {
    if (!projectPath) {
      sysLog('Set project path in Config tab first.');
      switchTab('config');
      return;
    }
    btnStart.disabled = true;
    setStatus('connecting');
    const result = await window.electronAPI.startTraining({
      pythonPath,
      projectPath,
      module: trainMode === 'ppo' ? 'server.ppo_train' : 'server.train',
    });
    if (result.ok) {
      serverRunning = true;
      btnStop.disabled = false;
      btnNextGen.disabled = false;
      btnEarlyNextGen.disabled = false;
      speedBtns.forEach(btn => { btn.disabled = false; });
      sysLog(`Training server started (PID ${result.pid})`);
      // Give Python a moment to bind the port, then connect WS
      setTimeout(connectWS, 1200);
    } else {
      btnStart.disabled = false;
      setStatus('error');
      sysLog(`Failed to start: ${result.error}`, 'err');
    }
  }

  async function stopServer() {
    if (ws) { try { ws.close(); } catch(_){} ws = null; }
    clearTimeout(reconnectTimer);
    await window.electronAPI.stopTraining();
    serverRunning  = false;
    wsConnected    = false;
    btnStart.disabled        = false;
    btnStop.disabled         = true;
    btnNextGen.disabled      = true;
    btnEarlyNextGen.disabled = true;
    speedBtns.forEach(btn => { btn.disabled = true; });
    setStatus('offline');
    sysLog('Training server stopped.');
  }

  function sendNextGen() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(msgpack.encode([MSG_NEXT_GEN]));
      sysLog('Sent Next Generation request.');
    }
  }

  // ── IPC from main process ──────────────────────────────────────────────────
  if (window.electronAPI) {
    window.electronAPI.onServerLog(({ level, text }) => {
      appendLog(text.trimEnd(), level === 'error' ? 'err' : 'info');
    });
    window.electronAPI.onServerStopped(({ code }) => {
      serverRunning = false;
      wsConnected   = false;
      btnStart.disabled        = false;
      btnStop.disabled         = true;
      btnNextGen.disabled      = true;
      btnEarlyNextGen.disabled = true;
      speedBtns.forEach(btn => { btn.disabled = true; });
      setStatus('offline');
      sysLog(`Server process exited (code ${code}).`);
    });
    window.electronAPI.onMenuStartTraining(startServer);
    window.electronAPI.onMenuStopTraining(stopServer);
    window.electronAPI.onMenuNextGen(sendNextGen);
  }

  // ── WebSocket ──────────────────────────────────────────────────────────────
  function connectWS() {
    if (ws) { try { ws.close(); } catch(_){} }
    setStatus('connecting');
    ws = new WebSocket(`ws://localhost:${wsPort}`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      wsConnected = true;
      setStatus('running');
      sysLog(`Connected to training server ws://localhost:${wsPort}`);
      if (Object.keys(ctrlValues).length) {
        ws.send(msgpack.encode([MSG_SET_TRAINING_PARAMS, {...ctrlValues}]));
      }
    };

    ws.onmessage = (evt) => {
      if (!(evt.data instanceof ArrayBuffer)) return;
      try {
        const msg  = msgpack.decode(new Uint8Array(evt.data));
        const type = msg[0];

        if (type === MSG_INIT) {
          State.init(msg);
          SpectateViewport.init(gameCanvas);
          wsLog(`MSG_INIT — world ${State.worldW}×${State.worldH}, tick_rate=${State.tickRate}`);

        } else if (type === MSG_TICK) {
          State.applyTick(msg);
          tpsTicks++;
          const now = performance.now();
          if (now - tpsStart >= 1000) {
            tps      = Math.round(tpsTicks * 1000 / (now - tpsStart));
            tpsTicks = 0;
            tpsStart = now;
          }

        } else if (type === MSG_TRAINING_STATS) {
          // Compute actual sim TPS from tick_num delta between consecutive stats packets
          const statsTick = msg[1] || 0;
          const nowMs = performance.now();
          if (lastStatsTick >= 0 && nowMs > lastStatsTime && statsTick > lastStatsTick) {
            simTps = Math.round((statsTick - lastStatsTick) * 1000 / (nowMs - lastStatsTime));
          }
          lastStatsTick = statsTick;
          lastStatsTime = nowMs;

          trainGen      = msg[2]  || 0;
          trainTimeLeft = msg[3]  || 0;
          trainPop      = msg[4]  || 0;
          trainTopFit   = msg[5]  || 0;
          trainAvgFit   = msg[6]  || 0;
          trainBestMass = msg[7]  || 0;
          trainAvgMass  = msg[8]  || 0;
          trainDeaths   = msg[9]  || 0;
          trainBots     = msg[10] || [];
          trainFood     = msg[11] || 0;
          if (msg[12] !== undefined) {
            const serverVal = Boolean(msg[12]);
            if (serverVal !== earlyNextGen) {
              earlyNextGen = serverVal;
              _syncEarlyNextGenButton();
            }
          }
          if (msg[13] !== undefined) {
            const serverScale = parseFloat(msg[13]);
            if (serverScale !== timeScale) {
              timeScale = serverScale;
              genTimeSec = (parseFloat($('tc-gen-time').value) || GENERATION_TIME) / timeScale;
              speedBtns.forEach(btn => {
                const active = parseFloat(btn.dataset.scale) === timeScale;
                btn.classList.toggle('active-toggle', active);
                btn.classList.toggle('inactive-toggle', !active);
              });
            }
          }
          if (msg[14] !== undefined) _syncControlsFromServer(msg[14]);
          _onStats();
        } else if (type === MSG_PPO_STATS) {
          // [MSG_PPO_STATS, rollout, env_steps, n_bots, avg_fitness,
          //  pg_loss, vf_loss, entropy, approx_kl, players_info, food_count]
          ppoRollout   = msg[1] || 0;
          ppoEnvSteps  = msg[2] || 0;
          trainPop     = msg[3] || 0;
          trainAvgFit  = msg[4] || 0;
          ppoPgLoss    = msg[5] || 0;
          ppoVfLoss    = msg[6] || 0;
          ppoEntropy   = msg[7] || 0;
          ppoKl        = msg[8] || 0;
          trainBots    = msg[9] || [];
          trainFood    = msg[10] || 0;
          _onPpoStats();

        }
      } catch(e) {
        wsLog(`decode error: ${e.message}`, 'err');
      }
    };

    ws.onclose = () => {
      wsConnected = false;
      if (serverRunning) {
        setStatus('connecting');
        reconnectTimer = setTimeout(connectWS, 2000);
      } else {
        setStatus('offline');
      }
    };
    ws.onerror = () => {};
  }

  // Try auto-connecting if server already running
  (async () => {
    if (window.electronAPI) {
      // Auto-detect project path: renderer/ is inside desktop/, which is inside project root
      // __dirname is not available; use a known relative path from the renderer file
      projectPath = '..';
      $('cfg-projpath').value = projectPath;
      $('cfg-python').value   = pythonPath;

      const r = await window.electronAPI.isTrainingRunning();
      if (r.running) {
        serverRunning = true;
        btnStart.disabled        = true;
        btnStop.disabled         = false;
        btnNextGen.disabled      = false;
        btnEarlyNextGen.disabled = false;
        speedBtns.forEach(btn => { btn.disabled = false; });
        connectWS();
      }
      loadConfigValues();
    }
  })();

  // ── Stats handler ──────────────────────────────────────────────────────────
  function _onStats() {
    playerSpeciesMap.clear();
    for (const b of trainBots) {
      if (b[8] !== undefined) playerSpeciesMap.set(b[0], b[8]);
    }
    window.playerSpeciesMap = playerSpeciesMap;
    _updateSidebar();
    _rebuildBotList();
    _recordHistory();
    if (activeTab === 'charts') refreshCharts();
  }

  function _onPpoStats() {
    // Update PPO panel
    if (ppoPanelRollout) ppoPanelRollout.textContent = ppoRollout.toLocaleString();
    if (ppoPanelSteps)   ppoPanelSteps.textContent   = ppoEnvSteps.toLocaleString();
    if (ppoPanelPg)      ppoPanelPg.textContent      = ppoPgLoss.toFixed(5);
    if (ppoPanelVf)      ppoPanelVf.textContent      = ppoVfLoss.toFixed(5);
    if (ppoPanelEnt)     ppoPanelEnt.textContent     = ppoEntropy.toFixed(4);
    if (ppoPanelKl)      ppoPanelKl.textContent      = ppoKl.toFixed(5);
    genBanner.textContent = `Rollout ${ppoRollout}  |  ${ppoEnvSteps.toLocaleString()} steps`;
    // Shared sidebar fields
    sSAvgfit.textContent = fmt(trainAvgFit);
    sSPop.textContent    = trainPop;
    sSFood.textContent   = trainFood.toLocaleString();
    _rebuildBotList();
  }

  function _updateSidebar() {
    sSgen.textContent    = trainGen;
    sStime.textContent   = trainTimeLeft.toFixed(1) + 's';
    sStimebar.style.width= Math.round((trainTimeLeft / genTimeSec) * 100) + '%';
    genBanner.textContent= `Gen ${trainGen}  |  ${trainTimeLeft.toFixed(0)}s`;
    sSTopfit.textContent = fmt(trainTopFit);
    sSAvgfit.textContent = fmt(trainAvgFit);
    sSBmass.textContent  = fmt(trainBestMass);
    sSAmass.textContent  = fmt(trainAvgMass);
    sSPop.textContent    = trainPop;
    sSDeaths.textContent = trainDeaths;
    sSFood.textContent   = trainFood.toLocaleString();

    if (followId) {
      const sorted = [...trainBots].sort((a, b) => b[2] - a[2]);
      const rank = sorted.findIndex(p => p[0] === followId) + 1;
      const p = trainBots.find(b => b[0] === followId);
      if (p) {
        fMass.textContent  = fmt(p[2]);
        fCells.textContent = p[5];
        fRank.textContent  = rank > 0 ? `#${rank}` : '—';
        fName.textContent  = p[1];
        followBadge.textContent = `Following: ${p[1]}`;
      }
    }
  }

  function setSelectedSpecies(sid) {
    selectedSpecies = sid;
    window.selectedSpecies = sid;
    _rebuildBotList();
  }

  function _rebuildBotList() {
    const byFitness = sortMode === 'fitness';
    const sorted    = [...trainBots].sort((a, b) =>
      byFitness ? (b[7] || 0) - (a[7] || 0) : b[2] - a[2]
    );
    const maxVal = sorted.length > 0
      ? (byFitness ? Math.max(sorted[0][7] || 0, 1) : sorted[0][2])
      : 1;
    botCount.textContent = `(${sorted.length})`;

    let html = '';
    for (const p of sorted) {
      const [id, name, mass, , , cells] = p;
      const fitness   = p[7] || 0;
      const speciesId = p[8] !== undefined ? p[8] : -1;
      const hue       = State.getHue(id) >= 0 ? State.getHue(id) : (id * 137.508) % 360;
      const color     = `hsl(${hue},70%,55%)`;
      const displayVal = byFitness ? fmt(fitness) : fmt(mass);
      const barVal     = byFitness ? fitness : mass;
      const barPct     = Math.max(2, Math.round((barVal / maxVal) * 100));
      const isFollow   = id === followId;
      const isSpeciesHl = selectedSpecies !== null && speciesId === selectedSpecies;
      const spSelected  = isSpeciesHl ? ' selected' : '';
      let rowClass = 'bot-row';
      if (isFollow)    rowClass += ' active';
      if (isSpeciesHl) rowClass += ' species-hl';
      const spLabel = speciesId >= 0 ? `S${speciesId}` : '??';
      html +=
        `<div class="${rowClass}" data-id="${id}" style="--bc:${color}">` +
        `<span class="br-icon">🤖</span>` +
        `<span class="br-species${spSelected}" data-species="${speciesId}">${spLabel}</span>` +
        `<span class="br-name">${esc(name)}</span>` +
        `<span class="br-mass">${displayVal}</span>` +
        `<div class="br-bar" style="width:${barPct}%"></div>` +
        `</div>`;
    }
    botList.innerHTML = html;
    botList.querySelectorAll('.bot-row').forEach(row => {
      row.addEventListener('click', () => {
        const id = parseInt(row.dataset.id, 10);
        setFollow(id === followId ? null : id);
      });
    });
    botList.querySelectorAll('.br-species').forEach(badge => {
      badge.addEventListener('click', (e) => {
        e.stopPropagation();
        const sid = parseInt(badge.dataset.species, 10);
        setSelectedSpecies(selectedSpecies === sid ? null : sid);
      });
    });
  }

  // ── History recording ──────────────────────────────────────────────────────
  function _recordHistory() {
    allTimeBestFit  = Math.max(allTimeBestFit,  trainTopFit);
    allTimeBestMass = Math.max(allTimeBestMass, trainBestMass);

    // Build species snapshot from current bot list
    const snap = {};
    for (const b of trainBots) {
      const sid = b[8] !== undefined ? b[8] : -1;
      if (!snap[sid]) snap[sid] = { count: 0, topFit: 0, totalFit: 0, bestMass: 0 };
      snap[sid].count++;
      snap[sid].topFit   = Math.max(snap[sid].topFit,  b[7] || 0);
      snap[sid].totalFit += b[7] || 0;
      snap[sid].bestMass = Math.max(snap[sid].bestMass, b[2] || 0);
    }
    for (const sid in snap) snap[sid].avgFit = snap[sid].totalFit / snap[sid].count;
    currentSpeciesSnapshot = snap;

    const last = history.generations[history.generations.length - 1];
    if (last !== trainGen) {
      history.generations.push(trainGen);
      history.topFitness.push(trainTopFit);
      history.avgFitness.push(trainAvgFit);
      history.bestMass.push(trainBestMass);
      history.avgMass.push(trainAvgMass);
      history.deaths.push(trainDeaths);
      history.popSize.push(trainPop);
      history.speciesSnapshots.push(snap);
      currentGenFitnesses = [];
    } else {
      const i = history.generations.length - 1;
      history.topFitness[i] = Math.max(history.topFitness[i] || 0, trainTopFit);
      history.avgFitness[i] = trainAvgFit;
      history.bestMass[i]   = Math.max(history.bestMass[i] || 0, trainBestMass);
      history.avgMass[i]    = trainAvgMass;
      history.deaths[i]     = trainDeaths;
      history.popSize[i]    = trainPop;
      if (i >= 0) history.speciesSnapshots[i] = snap;
    }
    currentGenFitnesses = trainBots.map(b => b[7] || 0);
    totalDeathsEver = history.deaths.reduce((s, v) => s + v, 0);
  }

  // ── Charts ─────────────────────────────────────────────────────────────────
  let charts = {};

  function _makeChart(id, cfg) {
    const existing = charts[id];
    if (existing) existing.destroy();
    const ctx = document.getElementById(id).getContext('2d');
    charts[id] = new Chart(ctx, cfg);
    return charts[id];
  }

  const CHART_DEFAULTS = {
    plugins: { legend: { labels: { color: 'rgba(255,255,255,0.6)', font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
      y: { ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
    },
    animation: { duration: 0 },
    responsive: true,
    maintainAspectRatio: false,
  };

  function mergeDeep(target, source) {
    for (const k of Object.keys(source)) {
      if (source[k] && typeof source[k] === 'object' && !Array.isArray(source[k])) {
        target[k] = target[k] || {};
        mergeDeep(target[k], source[k]);
      } else {
        target[k] = source[k];
      }
    }
    return target;
  }

  function speciesColor(sid, alpha = 1) {
    const hue = (sid * 137 + 37) % 360;
    return alpha < 1
      ? `hsla(${hue},65%,55%,${alpha})`
      : `hsl(${hue},65%,55%)`;
  }

  function _getRangedData() {
    const total = history.generations.length;
    const start = chartRange === 0 ? 0 : Math.max(0, total - chartRange);
    return {
      generations:      history.generations.slice(start),
      topFitness:       history.topFitness.slice(start),
      avgFitness:       history.avgFitness.slice(start),
      bestMass:         history.bestMass.slice(start),
      avgMass:          history.avgMass.slice(start),
      deaths:           history.deaths.slice(start),
      popSize:          history.popSize.slice(start),
      speciesSnapshots: history.speciesSnapshots.slice(start),
    };
  }

  // Update chart data in-place if dataset count matches, else recreate.
  function _applyChart(id, type, labels, datasets, extraOptions) {
    const ex = charts[id];
    if (ex && ex.data.datasets.length === datasets.length) {
      ex.data.labels = labels;
      datasets.forEach((ds, i) => { ex.data.datasets[i].data = ds.data; });
      ex.update('none');
    } else {
      if (ex) ex.destroy();
      const c2 = document.getElementById(id).getContext('2d');
      charts[id] = new Chart(c2, {
        type,
        data: { labels, datasets },
        options: mergeDeep(JSON.parse(JSON.stringify(CHART_DEFAULTS)), extraOptions || {}),
      });
    }
  }

  function refreshCharts() {
    const d    = _getRangedData();
    const gens = d.generations;

    // Update range label
    const ctRange = $('ct-gen-range');
    if (ctRange) {
      if (gens.length === 0) {
        ctRange.textContent = 'no data';
      } else if (gens.length === history.generations.length) {
        ctRange.textContent = `all (${gens.length})`;
      } else {
        ctRange.textContent = `${gens[0]}–${gens[gens.length - 1]}`;
      }
    }

    // Fitness line chart
    _applyChart('cvs-fitness', 'line', gens, [
      { label: 'Top Fitness', data: d.topFitness, borderColor: '#fa4', backgroundColor: 'rgba(255,170,0,0.1)', tension: 0.3, pointRadius: 2, borderWidth: 2, fill: true },
      { label: 'Avg Fitness', data: d.avgFitness, borderColor: '#4af', backgroundColor: 'rgba(68,170,255,0.06)', tension: 0.3, pointRadius: 2, borderWidth: 1.5, fill: true, borderDash: [4, 3] },
    ]);

    // Mass line chart
    _applyChart('cvs-mass', 'line', gens, [
      { label: 'Best Mass', data: d.bestMass, borderColor: '#6f6', backgroundColor: 'rgba(100,255,100,0.08)', tension: 0.3, pointRadius: 2, borderWidth: 2, fill: true },
      { label: 'Avg Mass',  data: d.avgMass,  borderColor: '#aaa', backgroundColor: 'rgba(170,170,170,0.05)', tension: 0.3, pointRadius: 2, borderWidth: 1.5, fill: false, borderDash: [4, 3] },
    ]);

    // Population + deaths bar chart
    _applyChart('cvs-pop', 'bar', gens, [
      { label: 'Bots Alive', data: d.popSize, backgroundColor: 'rgba(68,170,255,0.5)', borderColor: '#4af', borderWidth: 1 },
      { label: 'Deaths',     data: d.deaths,  backgroundColor: 'rgba(255,80,80,0.4)',  borderColor: '#f55', borderWidth: 1 },
    ]);

    // Fitness distribution histogram (current gen)
    const bins = 20;
    const vals = currentGenFitnesses;
    if (vals.length > 0) {
      const lo = Math.min(...vals), hi = Math.max(...vals) + 1;
      const step = (hi - lo) / bins;
      const counts = new Array(bins).fill(0);
      const distLabels = [];
      for (let i = 0; i < bins; i++) distLabels.push(fmt(lo + i * step));
      for (const v of vals) counts[Math.min(Math.floor((v - lo) / step), bins - 1)]++;
      _applyChart('cvs-dist', 'bar', distLabels, [
        { label: 'Bots', data: counts, backgroundColor: 'rgba(250,170,0,0.5)', borderColor: '#fa4', borderWidth: 1 },
      ]);
    }

    // Species stacked bar chart
    _refreshSpeciesChart(d);

    _buildSpeciesPanel();
    _updateChartSummary();
  }

  function _refreshSpeciesChart(d) {
    const snapshots = d.speciesSnapshots;
    const gens      = d.generations;
    if (!snapshots || snapshots.length === 0) return;

    const allSids = new Set();
    for (const snap of snapshots) for (const sid of Object.keys(snap)) allSids.add(Number(sid));
    const sidList = [...allSids].sort((a, b) => a - b);

    const datasets = sidList.map(sid => ({
      label:           `S${sid}`,
      data:            snapshots.map(s => (s[sid] ? s[sid].count : 0)),
      backgroundColor: speciesColor(sid, 0.75),
      borderColor:     speciesColor(sid),
      borderWidth:     1,
    }));

    _makeChart('cvs-species', {
      type: 'bar',
      data: { labels: gens, datasets },
      options: mergeDeep(JSON.parse(JSON.stringify(CHART_DEFAULTS)), {
        scales: {
          x: { stacked: true },
          y: { stacked: true },
        },
        plugins: {
          legend: {
            display: sidList.length <= 10,
            labels: { color: 'rgba(255,255,255,0.6)', font: { size: 10 }, boxWidth: 12 },
          },
        },
      }),
    });
  }

  function _buildSpeciesPanel() {
    const snap    = currentSpeciesSnapshot;
    const entries = Object.entries(snap).map(([sid, d]) => ({ sid: Number(sid), ...d }));
    entries.sort((a, b) => spSortMode === 'fitness' ? b.topFit - a.topFit : b.count - a.count);

    const totalBots  = entries.reduce((s, e) => s + e.count, 0);
    const spCountEl  = $('sp-count');
    if (spCountEl) spCountEl.textContent = `${entries.length} species`;

    const list = $('sp-list');
    if (!list) return;

    let html = '';
    for (const sp of entries) {
      const color = speciesColor(sp.sid);
      const isSelected = sp.sid === selectedSpecies;
      const pct = totalBots > 0 ? Math.round((sp.count / totalBots) * 100) : 0;
      html +=
        `<div class="sp-row${isSelected ? ' sp-selected' : ''}" data-sid="${sp.sid}">` +
        `<div class="sp-dot" style="background:${color}"></div>` +
        `<div class="sp-info">` +
          `<div style="display:flex;align-items:baseline;gap:5px">` +
            `<span class="sp-name">S${sp.sid}</span>` +
            `<span style="font-size:9px;color:var(--text-dim)">${pct}%</span>` +
          `</div>` +
          `<div class="sp-bar-wrap"><div class="sp-bar" style="width:${pct}%;background:${color}"></div></div>` +
        `</div>` +
        `<div class="sp-stat">` +
          `<span class="sp-stat-count">${sp.count}</span>` +
          `<span class="sp-stat-fit">${fmt(sp.topFit)}</span>` +
        `</div>` +
        `</div>`;
    }
    list.innerHTML = html;
    list.querySelectorAll('.sp-row').forEach(row => {
      row.addEventListener('click', () => {
        const sid = Number(row.dataset.sid);
        setSelectedSpecies(selectedSpecies === sid ? null : sid);
        _buildSpeciesPanel();
      });
    });
  }

  function _updateChartSummary() {
    scGen.textContent       = trainGen;
    scBestFit.textContent   = fmt(allTimeBestFit);
    scBestMass.textContent  = fmt(allTimeBestMass);
    scTotalDead.textContent = totalDeathsEver.toLocaleString();
  }

  // ── Config tab ─────────────────────────────────────────────────────────────
  $('cfg-projpath').addEventListener('change', (e) => { projectPath = e.target.value.trim(); });
  $('cfg-python').addEventListener('change',   (e) => { pythonPath  = e.target.value.trim(); });

  $('btn-cfg-detect').addEventListener('click', async () => {
    // Try '../' relative to renderer folder
    projectPath = '..';
    $('cfg-projpath').value = projectPath;
    await loadConfigValues();
  });

  $('btn-reload-cfg').addEventListener('click', loadConfigValues);

  async function loadConfigValues() {
    projectPath = $('cfg-projpath').value.trim() || projectPath;
    if (!projectPath) return;

    // Load server/training.py
    const tr = await window.electronAPI.readTrainingConfig(projectPath);
    if (tr.ok) {
      const txt = tr.text;
      $('tc-gen-time').value = _extractPyFloat(txt, 'GENERATION_TIME') ?? 60;
      $('tc-pop-size').value = _extractPyFloat(txt, 'TRAIN_POP_SIZE')  ?? 100;
      $('tc-survive') .value = _extractPyFloat(txt, 'SURVIVE_FRACTION')?? 0.5;
      $('tc-port')    .value = _extractPyFloat(txt, 'TRAIN_PORT')      ?? 8766;
      genTimeSec = parseFloat($('tc-gen-time').value) || GENERATION_TIME;
    }

    // Load server/config.py
    const cr = await window.electronAPI.readConfig(projectPath);
    if (cr.ok) {
      const txt = cr.text;
      $('wc-w')       .value = _extractPyFloat(txt, 'WORLD_W')       ?? 6000;
      $('wc-h')       .value = _extractPyFloat(txt, 'WORLD_H')       ?? 6000;
      $('wc-tick')    .value = _extractPyFloat(txt, 'TICK_RATE')      ?? 20;
      $('wc-food')    .value = _extractPyFloat(txt, 'FOOD_TARGET')    ?? 500;
      $('wc-virus')   .value = _extractPyFloat(txt, 'VIRUS_TARGET')   ?? 10;
      $('wc-speed')   .value = _extractPyFloat(txt, 'BASE_SPEED')     ?? 1200;
      $('wc-splitmass').value= _extractPyFloat(txt, 'MIN_SPLIT_MASS') ?? 35;
      $('wc-decay')   .value = _extractPyFloat(txt, 'DECAY_RATE')     ?? 0.01;
    }
  }

  function _extractPyFloat(src, name) {
    const m = src.match(new RegExp(`^\\s*${name}\\s*=\\s*([\\d.]+)`, 'm'));
    return m ? parseFloat(m[1]) : null;
  }

  function _setPyValue(src, name, value) {
    return src.replace(new RegExp(`(^\\s*${name}\\s*=\\s*)[\\d.]+`, 'm'), `$1${value}`);
  }

  $('btn-save-traincfg').addEventListener('click', async () => {
    const tr = await window.electronAPI.readTrainingConfig(projectPath);
    if (!tr.ok) { _cfgStatus('cfg-status-train', 'Error: ' + tr.error, false); return; }
    let txt = tr.text;
    txt = _setPyValue(txt, 'GENERATION_TIME',  $('tc-gen-time').value);
    txt = _setPyValue(txt, 'TRAIN_POP_SIZE',   $('tc-pop-size').value);
    txt = _setPyValue(txt, 'SURVIVE_FRACTION', $('tc-survive') .value);
    txt = _setPyValue(txt, 'TRAIN_PORT',       $('tc-port')    .value);
    const res = await window.electronAPI.writeConfig({ projectPath, text: txt, file: 'server/training.py' });
    _cfgStatus('cfg-status-train', res.ok ? 'Saved! Restart required.' : 'Error: ' + res.error, res.ok);
    genTimeSec = parseFloat($('tc-gen-time').value) || genTimeSec;
    wsPort     = parseInt($('tc-port').value, 10) || wsPort;
  });

  $('btn-save-worldcfg').addEventListener('click', async () => {
    const cr = await window.electronAPI.readConfig(projectPath);
    if (!cr.ok) { _cfgStatus('cfg-status-world', 'Error: ' + cr.error, false); return; }
    let txt = cr.text;
    txt = _setPyValue(txt, 'WORLD_W',       $('wc-w')       .value);
    txt = _setPyValue(txt, 'WORLD_H',       $('wc-h')       .value);
    txt = _setPyValue(txt, 'TICK_RATE',     $('wc-tick')    .value);
    txt = _setPyValue(txt, 'FOOD_TARGET',   $('wc-food')    .value);
    txt = _setPyValue(txt, 'VIRUS_TARGET',  $('wc-virus')   .value);
    txt = _setPyValue(txt, 'BASE_SPEED',    $('wc-speed')   .value);
    txt = _setPyValue(txt, 'MIN_SPLIT_MASS',$('wc-splitmass').value);
    txt = _setPyValue(txt, 'DECAY_RATE',    $('wc-decay')   .value);
    const res = await window.electronAPI.writeConfig({ projectPath, text: txt });
    _cfgStatus('cfg-status-world', res.ok ? 'Saved! Restart required.' : 'Error: ' + res.error, res.ok);
  });

  function _cfgStatus(id, msg, ok) {
    const el = $(id);
    el.textContent   = msg;
    el.style.color   = ok ? '#6f6' : '#f88';
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 4000);
  }

  // ── Logs ───────────────────────────────────────────────────────────────────
  let showServer = true;
  let showWs     = true;
  $('log-show-server').addEventListener('change', e => { showServer = e.target.checked; });
  $('log-show-ws')    .addEventListener('change', e => { showWs     = e.target.checked; });
  $('btn-clear-log')  .addEventListener('click', () => { logOut.innerHTML = ''; });

  $('btn-export-log').addEventListener('click', async () => {
    const res = await window.electronAPI.exportLog({
      projectPath,
      data: { history, allTimeBestFit, allTimeBestMass, totalDeathsEver, exportedAt: new Date().toISOString() },
    });
    sysLog(res.ok ? `Exported to: ${res.path}` : `Export failed: ${res.error}`);
  });

  function appendLog(text, cls = 'info') {
    const show = cls === 'sys' || (cls === 'err' && showServer) || (cls === 'info' && showServer) || (cls === 'ws' && showWs);
    if (!show) return;
    const line = document.createElement('div');
    line.className = 'log-line' + (cls !== 'info' ? ` ${cls}` : '');
    line.textContent = text;
    logOut.appendChild(line);
    if ($('log-autoscroll').checked) logOut.scrollTop = logOut.scrollHeight;
  }
  function sysLog(msg)  { appendLog(`[sys] ${msg}`, 'sys'); }
  function wsLog(msg, cls = 'ws') { if (showWs) appendLog(`[ws] ${msg}`, cls); }

  // ── Status helpers ─────────────────────────────────────────────────────────
  function setStatus(state) {
    statusDot.className = '';
    const map = { running: ['running', 'Connected'], connecting: ['connecting', 'Connecting…'], error: ['error', 'Error'], offline: ['', 'Offline'] };
    const [cls, txt] = map[state] || ['', state];
    if (cls) statusDot.classList.add(cls);
    statusText.textContent = txt;
  }

  // ── Format helpers ─────────────────────────────────────────────────────────
  function fmt(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 1000)      return (n / 1000).toFixed(1) + 'k';
    return Math.floor(n).toString();
  }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') { e.preventDefault(); sendNextGen(); }
    if (e.key === '+' || e.key === '=') applyZoom(zoomIdx + 1);
    if (e.key === '-')                  applyZoom(zoomIdx - 1);
    if (e.key === '0')                  { applyZoom(3); setFollow(null); }
    if (e.key === 'Escape')             { setFollow(null); setSelectedSpecies(null); }
    if (e.key === '1') switchTab('viewer');
    if (e.key === '2') switchTab('charts');
    if (e.key === '3') switchTab('config');
    if (e.key === '4') switchTab('controls');
    if (e.key === '5') switchTab('logs');
  });

  // ── Charts controls ────────────────────────────────────────────────────────
  document.querySelectorAll('.ct-range-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      chartRange = parseInt(btn.dataset.range, 10);
      document.querySelectorAll('.ct-range-btn').forEach(b =>
        b.classList.toggle('ct-active', b === btn));
      if (activeTab === 'charts') refreshCharts();
    });
  });

  const spSortBtn = $('sp-sort-btn');
  if (spSortBtn) {
    spSortBtn.addEventListener('click', () => {
      spSortMode = spSortMode === 'count' ? 'fitness' : 'count';
      spSortBtn.textContent = spSortMode === 'fitness' ? 'Fitness ↓' : 'Count ↓';
      _buildSpeciesPanel();
    });
  }

  // ── Controls persistence ───────────────────────────────────────────────────
  function _loadCtrlPersisted() {
    try {
      const raw = localStorage.getItem('rogue.ctrlValues');
      if (raw) Object.assign(ctrlValues, JSON.parse(raw));
    } catch(e) {}
  }

  function _saveCtrlPersisted() {
    try { localStorage.setItem('rogue.ctrlValues', JSON.stringify(ctrlValues)); } catch(e) {}
  }

  // ── Controls tab ───────────────────────────────────────────────────────────
  function _initControlsTab() {
    if (_ctrlBuilt) return;
    _ctrlBuilt = true;
    const scroll = $('ctrl-scroll');
    if (!scroll) return;

    for (const group of CTRL_DEFS) {
      const card = document.createElement('div');
      card.className = 'ctrl-card';
      const h = document.createElement('h3');
      h.textContent = group.group;
      card.appendChild(h);

      for (const item of group.items) {
        const val = ctrlValues[item.key] ?? item.def;
        const dirty = ctrlDirty.has(item.key);

        const row = document.createElement('div');
        row.className = 'ctrl-row' + (dirty ? ' dirty' : '');
        row.dataset.key = item.key;

        const lbl = document.createElement('label');
        lbl.textContent = item.label;
        lbl.title = `Default: ${item.def}`;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = item.min; slider.max = item.max; slider.step = item.step;
        slider.value = val;
        slider.className = 'ctrl-slider';

        const num = document.createElement('input');
        num.type = 'number';
        num.min = item.min; num.max = item.max; num.step = item.step;
        num.value = val;
        num.className = 'ctrl-num' + (dirty ? ' dirty' : '');

        slider.addEventListener('input', () => {
          num.value = slider.value;
          _onCtrlChange(item, parseFloat(slider.value));
        });
        num.addEventListener('change', () => {
          const v = Math.max(item.min, Math.min(item.max, parseFloat(num.value) || item.def));
          num.value = v; slider.value = v;
          _onCtrlChange(item, v);
        });

        row.append(lbl, slider, num);
        card.appendChild(row);
      }
      scroll.appendChild(card);
    }
  }

  function _onCtrlChange(item, value) {
    ctrlValues[item.key] = value;
    ctrlDirty.add(item.key);
    const row = document.querySelector(`.ctrl-row[data-key="${item.key}"]`);
    if (row) {
      row.classList.add('dirty');
      row.querySelector('.ctrl-num')?.classList.add('dirty');
    }
    _updateCtrlBadge();
    _saveCtrlPersisted();
    clearTimeout(_ctrlTimer);
    _ctrlTimer = setTimeout(_sendCtrlParams, 400);
  }

  function _sendCtrlParams() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const params = {};
    for (const key of ctrlDirty) params[key] = ctrlValues[key];
    if (!Object.keys(params).length) return;
    ws.send(msgpack.encode([MSG_SET_TRAINING_PARAMS, params]));
    sysLog(`[ctrl] Queued for next gen: ${JSON.stringify(params)}`);
  }

  function _syncControlsFromServer(params) {
    if (!params || typeof params !== 'object') return;
    for (const group of CTRL_DEFS) {
      for (const item of group.items) {
        const sv = params[item.key];
        if (sv === undefined) continue;
        if (ctrlDirty.has(item.key)) {
          // Clear dirty if server has now applied our value
          if (Math.abs(sv - (ctrlValues[item.key] ?? NaN)) < 1e-9) {
            ctrlDirty.delete(item.key);
            const row = document.querySelector(`.ctrl-row[data-key="${item.key}"]`);
            if (row) {
              row.classList.remove('dirty');
              row.querySelector('.ctrl-num')?.classList.remove('dirty');
            }
          }
          continue; // don't overwrite pending user value
        }
        ctrlValues[item.key] = sv;
        const row = document.querySelector(`.ctrl-row[data-key="${item.key}"]`);
        if (row) {
          const sl = row.querySelector('.ctrl-slider');
          const nm = row.querySelector('.ctrl-num');
          if (sl) sl.value = sv;
          if (nm) nm.value = sv;
        }
      }
    }
    _updateCtrlBadge();
  }

  function _updateCtrlBadge() {
    const badge = $('ctrl-tab-badge');
    if (!badge) return;
    if (ctrlDirty.size > 0) {
      badge.textContent = ctrlDirty.size;
      badge.style.display = 'inline-block';
    } else {
      badge.style.display = 'none';
    }
  }

  // ── Render loop ────────────────────────────────────────────────────────────
  let prevTs = 0;
  function gameLoop(ts) {
    const dt = prevTs > 0 ? ts - prevTs : 16.67;
    prevTs = ts;

    if (State.ready && activeTab === 'viewer') {
      State.tickFood(dt / 1000);
      const alpha = Math.min((performance.now() - State.lastTickTime) / State.smoothTickMs, 1.0);
      let cells    = [];
      let fallback = null;
      if (followId !== null) {
        for (const [, c] of State.cells) {
          if (c.playerId === followId) cells.push(c);
        }
        if (cells.length === 0) {
          fallback = trainBots.find(b => b[0] === followId) || null;
        }
      }
      SpectateViewport.update(cells, dt, fallback);
      Renderer.render(SpectateViewport, alpha);
      Minimap.render(SpectateViewport);
    }

    tpsDisplay.textContent = wsConnected
      ? (simTps > 0 ? `${simTps} sim TPS` : `${tps} TPS`)
      : '— TPS';
    requestAnimationFrame(gameLoop);
  }

  _loadCtrlPersisted();
  requestAnimationFrame(gameLoop);
  _syncEarlyNextGenButton();
  setStatus('offline');
  sysLog('rogue.io Training Desktop — ready. Set project path in Config and click Start Training.');

})();
