/**
 * train.js — Training viewer client
 *
 * Connects to the training server (ws://host:8766).
 * Handles:
 *   MSG_INIT           (0x10) — init world via State.init()
 *   MSG_TICK           (0x11) — apply tick via State.applyTick()
 *   MSG_TRAINING_STATS (0x24) — update training HUD
 *
 * [MSG_TRAINING_STATS, tick_num, generation, time_remaining, pop_size,
 *  top_fitness, avg_fitness, best_mass, avg_mass, total_deaths,
 *  players_info, total_food]
 *
 * players_info: [[id, name, mass, cx, cy, cell_count, is_bot], ...]
 */
(() => {
  'use strict';

  const MSG_INIT           = 0x10;
  const MSG_TICK           = 0x11;
  const MSG_TRAINING_STATS = 0x24;
  const MSG_NEXT_GEN       = 0x25;

  const ZOOM_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0];
  const ZOOM_LABELS      = ['0.25×', '0.5×', '0.75×', '1×', '1.5×', '2.5×', '4×'];
  const TRAIN_PORT       = 8766;
  const GENERATION_TIME  = 60.0; // must match server

  // ---- State ----
  let ws         = null;
  let followId   = null;
  let zoomIdx    = 3;

  // Training HUD data (from MSG_TRAINING_STATS)
  let trainGeneration  = 0;
  let trainTimeLeft    = 0;
  let trainPopSize     = 0;
  let trainTopFit      = 0;
  let trainAvgFit      = 0;
  let trainBestMass    = 0;
  let trainAvgMass     = 0;
  let trainDeaths      = 0;
  let trainPlayersInfo = [];  // [[id, name, mass, cx, cy, cells, is_bot], ...]
  let trainTotalFood   = 0;

  // TPS tracking
  let tpsTickCount = 0;
  let tpsStart     = performance.now();
  let displayTps   = 0;

  // ---- DOM refs ----
  const canvas       = document.getElementById('game');
  const minimapCanvas = document.getElementById('minimap');
  const genNum       = document.getElementById('gen-num');
  const genTime      = document.getElementById('gen-time');
  const timeBar      = document.getElementById('time-bar');
  const genBanner    = document.getElementById('gen-banner');
  const topFitnessEl = document.getElementById('top-fitness');
  const avgFitnessEl = document.getElementById('avg-fitness');
  const bestMassEl   = document.getElementById('best-mass');
  const avgMassEl    = document.getElementById('avg-mass');
  const popSizeEl    = document.getElementById('pop-size');
  const genDeathsEl  = document.getElementById('gen-deaths');
  const statsTpsEl   = document.getElementById('stats-tps');
  const statsFoodEl  = document.getElementById('stats-food');
  const followLabel  = document.getElementById('follow-label');
  const followMass   = document.getElementById('follow-mass');
  const followCells  = document.getElementById('follow-cells');
  const followRank   = document.getElementById('follow-rank');
  const followStats  = document.getElementById('follow-stats');
  const followBadge  = document.getElementById('follow-badge');
  const playerListEl = document.getElementById('player-list');
  const playerCount  = document.getElementById('player-count');
  const zoomLabel    = document.getElementById('zoom-label');
  const zoomInBtn    = document.getElementById('zoom-in');
  const zoomOutBtn   = document.getElementById('zoom-out');
  const nextGenBtn   = document.getElementById('next-gen-btn');

  // ---- Init subsystems ----
  Renderer.init(canvas);
  Minimap.init(minimapCanvas);
  SpectateViewport.init(canvas);

  function resize() { Renderer.resizeCanvas(); }
  window.addEventListener('resize', resize);
  resize();

  // ---- Zoom ----
  function applyZoom(idx) {
    zoomIdx = Math.max(0, Math.min(ZOOM_MULTIPLIERS.length - 1, idx));
    SpectateViewport.setZoomMultiplier(ZOOM_MULTIPLIERS[zoomIdx]);
    if (zoomLabel) zoomLabel.textContent = ZOOM_LABELS[zoomIdx];
  }

  if (zoomInBtn)  zoomInBtn.addEventListener('click',  () => applyZoom(zoomIdx + 1));
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => applyZoom(zoomIdx - 1));
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    applyZoom(e.deltaY < 0 ? zoomIdx + 1 : zoomIdx - 1);
  }, { passive: false });

  // ---- Next Generation Button ----
  if (nextGenBtn) {
    nextGenBtn.addEventListener('click', () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(msgpack.encode([MSG_NEXT_GEN]));
      }
    });
  }

  // ---- Follow ----
  function followBot(id) {
    followId = id || null;
    _updateFollowUI();
    _rebuildBotList();
  }

  function _updateFollowUI() {
    if (followId) {
      const p = trainPlayersInfo.find(e => e[0] === followId);
      const name = p ? p[1] : `Bot #${followId}`;
      if (followLabel) followLabel.textContent = name;
      if (followBadge) followBadge.textContent = `Following: ${name}`;
      if (followStats) followStats.style.display = '';
    } else {
      if (followLabel) followLabel.textContent = '— Overview —';
      if (followBadge) followBadge.textContent = 'Overview';
      if (followStats) followStats.style.display = 'none';
    }
  }

  // ---- Network ----
  function connect() {
    const host = `ws://${window.location.hostname}:${TRAIN_PORT}`;
    ws = new WebSocket(host);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {};

    ws.onmessage = (evt) => {
      const data = evt.data;
      if (!(data instanceof ArrayBuffer)) return;
      try {
        const msg  = msgpack.decode(new Uint8Array(data));
        const type = msg[0];

        if (type === MSG_INIT) {
          State.init(msg);
          SpectateViewport.init(canvas);

        } else if (type === MSG_TICK) {
          State.applyTick(msg);
          tpsTickCount++;
          const now = performance.now();
          if (now - tpsStart >= 1000) {
            displayTps   = Math.round(tpsTickCount * 1000 / (now - tpsStart));
            tpsTickCount = 0;
            tpsStart     = now;
          }

        } else if (type === MSG_TRAINING_STATS) {
          // [MSG_TRAINING_STATS, tick_num, generation, time_remaining, pop_size,
          //  top_fitness, avg_fitness, best_mass, avg_mass, total_deaths,
          //  players_info, total_food]
          trainGeneration  = msg[2]  || 0;
          trainTimeLeft    = msg[3]  || 0;
          trainPopSize     = msg[4]  || 0;
          trainTopFit      = msg[5]  || 0;
          trainAvgFit      = msg[6]  || 0;
          trainBestMass    = msg[7]  || 0;
          trainAvgMass     = msg[8]  || 0;
          trainDeaths      = msg[9]  || 0;
          trainPlayersInfo = msg[10] || [];
          trainTotalFood   = msg[11] || 0;
          console.log('Training stats received:', { generation: trainGeneration, bots: trainPlayersInfo.length });
          _updateHUD();
        }
      } catch (e) {
        console.warn('train decode error', e);
      }
    };

    ws.onclose = () => { setTimeout(connect, 2000); };
    ws.onerror = () => {};
  }

  // ---- HUD updates ----
  function _fmtNum(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1000)      return (n / 1000).toFixed(1) + 'k';
    return Math.floor(n).toString();
  }

  function _escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function _updateHUD() {
    if (genNum)       genNum.textContent       = trainGeneration;
    if (genTime)      genTime.textContent      = trainTimeLeft.toFixed(1) + 's';
    if (timeBar)      timeBar.style.width      = Math.round((trainTimeLeft / GENERATION_TIME) * 100) + '%';
    if (genBanner)    genBanner.textContent    = `Gen ${trainGeneration}  |  ${trainTimeLeft.toFixed(0)}s`;
    if (topFitnessEl) topFitnessEl.textContent = _fmtNum(trainTopFit);
    if (avgFitnessEl) avgFitnessEl.textContent = _fmtNum(trainAvgFit);
    if (bestMassEl)   bestMassEl.textContent   = _fmtNum(trainBestMass);
    if (avgMassEl)    avgMassEl.textContent    = _fmtNum(trainAvgMass);
    if (popSizeEl)    popSizeEl.textContent    = trainPopSize;
    if (genDeathsEl)  genDeathsEl.textContent  = trainDeaths;
    if (statsFoodEl)  statsFoodEl.textContent  = trainTotalFood.toLocaleString();

    // Update follow panel stats
    if (followId) {
      const sorted = [...trainPlayersInfo].sort((a, b) => b[5+1] - a[5+1]); // sort by fitness if available; fallback to mass
      const sortedByMass = [...trainPlayersInfo].sort((a, b) => b[2] - a[2]);
      const rank = sortedByMass.findIndex(p => p[0] === followId) + 1;
      const p    = trainPlayersInfo.find(e => e[0] === followId);
      if (p) {
        if (followMass)  followMass.textContent  = _fmtNum(p[2]);
        if (followCells) followCells.textContent = p[5];
        if (followRank)  followRank.textContent  = rank > 0 ? `#${rank}` : '—';
        if (followBadge) followBadge.textContent = `Following: ${p[1]}`;
        if (followLabel) followLabel.textContent = p[1];
      }
    }

    _rebuildBotList();
  }

  function _rebuildBotList() {
    if (!playerListEl) return;

    // Sort by mass (best first)
    const sorted  = [...trainPlayersInfo].sort((a, b) => b[2] - a[2]);
    const maxMass = sorted.length > 0 ? sorted[0][2] : 1;
    if (playerCount) playerCount.textContent = `(${sorted.length})`;

    console.log('Rebuilding bot list:', sorted.length, 'bots');

    let html = '';
    for (const p of sorted) {
      const [id, name, mass, , , cells] = p;
      const isFollowed = id === followId;
      const rawHue = State.getHue(id);
      const hue    = rawHue >= 0 ? rawHue : (id * 137.508) % 360;
      const color  = `hsl(${hue}, 70%, 55%)`;
      const barPct = Math.max(2, Math.round((mass / maxMass) * 100));

      html += `<div class="player-row${isFollowed ? ' followed' : ''}" data-id="${id}" style="--player-color:${color}">` +
        `<span class="p-icon">🤖</span>` +
        `<span class="p-name">${_escHtml(name)}</span>` +
        `<span class="p-mass">${_fmtNum(mass)}</span>` +
        `<div class="p-bar" style="width:${barPct}%"></div>` +
        `</div>`;
    }
    playerListEl.innerHTML = html;

    for (const row of playerListEl.querySelectorAll('.player-row')) {
      row.addEventListener('click', () => {
        const id = parseInt(row.dataset.id, 10);
        followBot(id === followId ? null : id);
      });
    }
  }

  // ---- Render loop ----
  let prevTimestamp = 0;

  function gameLoop(timestamp) {
    const dt = prevTimestamp > 0 ? timestamp - prevTimestamp : 16.67;
    prevTimestamp = timestamp;

    if (State.ready) {
      const alpha = Math.min(
        (performance.now() - State.lastTickTime) / State.smoothTickMs,
        1.0
      );

      let cells    = [];
      let fallback = null;
      if (followId !== null) {
        for (const [, c] of State.cells) {
          if (c.playerId === followId) cells.push(c);
        }
        if (cells.length === 0) {
          fallback = trainPlayersInfo.find(p => p[0] === followId) || null;
        }
      }

      SpectateViewport.update(cells, dt, fallback);
      Renderer.render(SpectateViewport, alpha);
      Minimap.render(SpectateViewport);
    }

    if (statsTpsEl) statsTpsEl.textContent = displayTps;

    requestAnimationFrame(gameLoop);
  }

  // ---- Boot ----
  applyZoom(3);
  connect();
  requestAnimationFrame(gameLoop);
})();