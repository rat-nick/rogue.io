/**
 * spectate.js — main logic for the spectator client.
 *
 * Connects to the game server with "SPECTATE" handshake, receives the normal
 * MSG_TICK stream (culled to the followed player's viewport) plus periodic
 * MSG_STATS snapshots.  The sidebar is driven entirely by MSG_STATS; rendering
 * reuses renderer.js / minimap.js unchanged.
 */
(() => {
  // ---- Protocol constants (must match server/protocol.py) ----
  const MSG_INIT   = 0x10;
  const MSG_TICK   = 0x11;
  const MSG_STATS  = 0x22;
  const MSG_FOLLOW = 0x23;

  // Zoom: viewport-size multipliers.  Higher = more zoomed IN.
  // Index 3 (1×) is the default — same scale as a normal player would see.
  const ZOOM_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0];
  const ZOOM_LABELS      = ['0.25×', '0.5×', '0.75×', '1×', '1.5×', '2.5×', '4×'];

  // ---- State ----
  let ws           = null;
  let followId     = null;   // null = overview; number = player ID being followed
  let zoomIdx      = 3;      // index into ZOOM_MULTIPLIERS (3 = 1×)
  let statsData    = [];     // [[id, name, mass, cx, cy, cells, is_bot], ...]
  let totalFood    = 0;
  let lastStatsTick = 0;

  // TPS tracking (client-side measurement of incoming ticks)
  let tpsTickCount = 0;
  let tpsStart     = performance.now();
  let displayTps   = 0;

  // ---- DOM refs ----
  const canvas        = document.getElementById('game');
  const minimapCanvas = document.getElementById('minimap');
  const statsFood     = document.getElementById('stats-food');
  const statsBots     = document.getElementById('stats-bots');
  const statsPlayers  = document.getElementById('stats-players');
  const statsTps      = document.getElementById('stats-tps');
  const statsTick     = document.getElementById('stats-tick');
  const followLabel   = document.getElementById('follow-label');
  const followMass    = document.getElementById('follow-mass');
  const followCells   = document.getElementById('follow-cells');
  const followRank    = document.getElementById('follow-rank');
  const followStats   = document.getElementById('follow-stats');
  const playerListEl  = document.getElementById('player-list');
  const playerCount   = document.getElementById('player-count');
  const zoomLabel     = document.getElementById('zoom-label');
  const zoomInBtn     = document.getElementById('zoom-in');
  const zoomOutBtn    = document.getElementById('zoom-out');
  const followBadge   = document.getElementById('follow-badge');

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
    sendFollowCmd();
  }

  if (zoomInBtn)  zoomInBtn.addEventListener('click',  () => applyZoom(zoomIdx + 1));
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => applyZoom(zoomIdx - 1));

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    applyZoom(e.deltaY < 0 ? zoomIdx + 1 : zoomIdx - 1);
  }, { passive: false });

  // ---- Follow commands ----
  function sendFollowCmd() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const buf  = new ArrayBuffer(6);
    const view = new DataView(buf);
    view.setUint8(0,  MSG_FOLLOW);
    view.setUint32(1, followId || 0, false);   // big-endian; 0 = overview
    view.setUint8(5,  zoomIdx);
    ws.send(buf);
  }

  function followPlayer(id) {
    followId = id || null;
    sendFollowCmd();
    _updateFollowUI();
    _rebuildPlayerList();
  }

  function _updateFollowUI() {
    if (followId) {
      const p = statsData.find(e => e[0] === followId);
      const name = p ? p[1] : `#${followId}`;
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
    const host = `ws://${window.location.hostname}:8765`;
    ws = new WebSocket(host);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      ws.send('SPECTATE');
      // Re-send follow command in case we reconnected mid-session
      if (followId !== null) sendFollowCmd();
    };

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
          // Track TPS
          tpsTickCount++;
          const now = performance.now();
          if (now - tpsStart >= 1000) {
            displayTps   = Math.round(tpsTickCount * 1000 / (now - tpsStart));
            tpsTickCount = 0;
            tpsStart     = now;
          }

        } else if (type === MSG_STATS) {
          // [MSG_STATS, tick_num, [[id,name,mass,cx,cy,cells,is_bot],...], total_food]
          lastStatsTick = msg[1];
          statsData     = msg[2] || [];
          totalFood     = msg[3] || 0;
          _updateSidebar();
        }
      } catch (e) {
        console.warn('spectate decode error', e);
      }
    };

    ws.onclose = () => { setTimeout(connect, 2000); };
    ws.onerror = () => {};
  }

  // ---- Sidebar updates ----
  function _escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function _fmtMass(m) {
    if (m >= 1_000_000) return (m / 1_000_000).toFixed(1) + 'M';
    if (m >= 1000)      return (m / 1000).toFixed(1) + 'k';
    return Math.floor(m).toString();
  }

  function _updateSidebar() {
    const bots   = statsData.filter(p => p[6] === 1);
    const humans = statsData.filter(p => p[6] === 0);

    if (statsFood)    statsFood.textContent    = totalFood.toLocaleString();
    if (statsBots)    statsBots.textContent    = bots.length;
    if (statsPlayers) statsPlayers.textContent = humans.length;
    if (statsTick)    statsTick.textContent    = lastStatsTick;

    // Update follow panel with current stats
    if (followId) {
      const sorted = [...statsData].sort((a, b) => b[2] - a[2]);
      const rank   = sorted.findIndex(p => p[0] === followId) + 1;
      const p      = statsData.find(e => e[0] === followId);
      if (p) {
        if (followMass)  followMass.textContent  = _fmtMass(p[2]);
        if (followCells) followCells.textContent = p[5];
        if (followRank)  followRank.textContent  = rank > 0 ? `#${rank}` : '—';
        // Keep badge name fresh
        if (followBadge) followBadge.textContent = `Following: ${p[1]}`;
        if (followLabel) followLabel.textContent = p[1];
      }
    }

    _rebuildPlayerList();
  }

  function _rebuildPlayerList() {
    if (!playerListEl) return;

    const sorted   = [...statsData].sort((a, b) => b[2] - a[2]);
    const maxMass  = sorted.length > 0 ? sorted[0][2] : 1;
    if (playerCount) playerCount.textContent = `(${sorted.length})`;

    let html = '';
    for (const p of sorted) {
      const [id, name, mass, , , cells, isBot] = p;
      const isFollowed = id === followId;
      const icon       = isBot ? '🤖' : '👤';
      const hue        = (id * 137.508) % 360;
      const color      = `hsl(${hue}, 70%, 55%)`;
      const barPct     = Math.max(2, Math.round((mass / maxMass) * 100));

      html += `<div class="player-row${isFollowed ? ' followed' : ''}" data-id="${id}" style="--player-color:${color}">` +
        `<span class="p-icon">${icon}</span>` +
        `<span class="p-name">${_escHtml(name)}</span>` +
        `<span class="p-cells">${cells}c</span>` +
        `<span class="p-mass">${_fmtMass(mass)}</span>` +
        `<div class="p-bar" style="width:${barPct}%"></div>` +
        `</div>`;
    }
    playerListEl.innerHTML = html;

    for (const row of playerListEl.querySelectorAll('.player-row')) {
      row.addEventListener('click', () => {
        const id = parseInt(row.dataset.id, 10);
        followPlayer(id === followId ? null : id);
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

      // Find cells for the followed player (null → overview)
      let cells    = [];
      let fallback = null;
      if (followId !== null) {
        for (const [, c] of State.cells) {
          if (c.playerId === followId) cells.push(c);
        }
        // If no cells yet, give viewport the last known stats position
        if (cells.length === 0) {
          fallback = statsData.find(p => p[0] === followId) || null;
        }
      }

      SpectateViewport.update(cells, dt, fallback);
      Renderer.render(SpectateViewport, alpha);
      Minimap.render(SpectateViewport);
    }

    // Refresh TPS counter in DOM every frame (cheap string write)
    if (statsTps) statsTps.textContent = displayTps;

    requestAnimationFrame(gameLoop);
  }

  // ---- Boot ----
  applyZoom(3);   // sets initial zoom label
  connect();
  requestAnimationFrame(gameLoop);
})();
