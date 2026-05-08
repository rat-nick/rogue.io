(() => {
  const TICK_MS = 1000 / 20; // 50ms - matches server TICK_RATE=20

  let animFrameId = null;
  let gameRunning = false;

  // DOM refs
  const gameCanvas = document.getElementById('game');
  const minimapCanvas = document.getElementById('minimap');
  const loginScreen = document.getElementById('login-screen');
  const deathScreen = document.getElementById('death-screen');
  const nameInput = document.getElementById('name-input');
  const playBtn = document.getElementById('play-btn');
  const respawnBtn = document.getElementById('respawn-btn');
  const deathScore = document.getElementById('death-score');
  const lbContainer = document.getElementById('leaderboard');

  // Init subsystems
  Renderer.init(gameCanvas);
  Minimap.init(minimapCanvas);
  Viewport.init(gameCanvas);
  Input.init();
  Leaderboard.init(lbContainer);

  // Resize canvas to fill window
  function resize() {
    Renderer.resizeCanvas();
  }
  window.addEventListener('resize', resize);
  resize();

  // Input send loop (matches server tick rate)
  let inputInterval = null;
  function startInputLoop() {
    if (inputInterval) clearInterval(inputInterval);
    inputInterval = setInterval(() => {
      if (!Network.connected) return;
      Input.updateMouseWorld();
      Network.sendInput(
        Input.mouseWorldX,
        Input.mouseWorldY,
        Input.consumeSplit(),
        Input.consumeEject()
      );
    }, TICK_MS);
  }

  // Main render loop
  let prevTimestamp = 0;
  function gameLoop(timestamp) {
    if (!gameRunning) return;

    const dt = prevTimestamp > 0 ? timestamp - prevTimestamp : 16.67;
    prevTimestamp = timestamp;
    const alpha = Math.min((performance.now() - State.lastTickTime) / State.smoothTickMs, 1.0);
    const ownCells = State.getOwnCells();

    State.updateMovingFood(dt / 1000);
    Viewport.update(ownCells, dt);
    Renderer.render(Viewport, alpha);
    Minimap.render(Viewport);
    Leaderboard.update();

    animFrameId = requestAnimationFrame(gameLoop);
  }

  function startGame() {
    gameRunning = true;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(gameLoop);
  }

  function stopGame() {
    gameRunning = false;
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  // Death handler
  function onDead(score, killerName) {
    stopGame();
    if (deathScore) {
      deathScore.textContent = `Score: ${Math.floor(score)}${killerName ? ` - eaten by ${killerName}` : ''}`;
    }
    if (deathScreen) deathScreen.style.display = 'flex';
  }

  // Play button
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      const name = (nameInput ? nameInput.value.trim() : '') || 'Player';
      const host = `ws://${window.location.hostname}:8765`;
      Network.connect(host, name, onDead);

      // Wait for state to be ready, then start
      const waitReady = setInterval(() => {
        if (State.ready) {
          clearInterval(waitReady);
          if (loginScreen) loginScreen.style.display = 'none';
          startInputLoop();
          startGame();
        }
      }, 50);
    });
  }

  // Respawn button
  if (respawnBtn) {
    respawnBtn.addEventListener('click', () => {
      if (deathScreen) deathScreen.style.display = 'none';
      State.reset();
      // The server will respawn us automatically;
      // but we need to reconnect since the server-side player was removed
      Network.disconnect();
      const name = (nameInput ? nameInput.value.trim() : '') || 'Player';
      const host = `ws://${window.location.hostname}:8765`;
      Network.connect(host, name, onDead);

      const waitReady = setInterval(() => {
        if (State.ready) {
          clearInterval(waitReady);
          startInputLoop();
          startGame();
        }
      }, 50);
    });
  }

  // Also allow Enter key on name input
  if (nameInput) {
    nameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') playBtn && playBtn.click();
    });
  }
})();
