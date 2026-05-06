const Renderer = (() => {
  let canvas = null;
  let ctx    = null;

  // Food color palette matching server config.FOOD_COLORS
  const FOOD_COLORS = [
    '#ff5555', '#ffa500', '#ffff55', '#55ff55',
    '#55ffff', '#5555ff', '#ff55ff', '#ffc8c8',
    '#c8ffc8', '#c8c8ff', '#ffffc8', '#c8e6ff',
  ];

  // Map player_id -> HSL color string (cached)
  const playerColors = new Map();
  function getPlayerColor(playerId) {
    if (!playerColors.has(playerId)) {
      const hue = (playerId * 137.508) % 360;   // golden angle distribution
      playerColors.set(playerId, `hsl(${hue}, 70%, 55%)`);
    }
    return playerColors.get(playerId);
  }

  function init(c) {
    canvas = c;
    ctx    = c.getContext('2d');
  }

  function resizeCanvas() {
    if (!canvas) return;
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  // Draw background grid lines (only within viewport)
  function drawGrid(vp) {
    const GRID_SPACING = 200;
    const rect = vp.getVisibleRect();

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth   = 1 / vp.scale;

    const startX = Math.floor(rect.x / GRID_SPACING) * GRID_SPACING;
    const startY = Math.floor(rect.y / GRID_SPACING) * GRID_SPACING;

    ctx.beginPath();
    for (let x = startX; x < rect.x + rect.w + GRID_SPACING; x += GRID_SPACING) {
      ctx.moveTo(x, rect.y);
      ctx.lineTo(x, rect.y + rect.h);
    }
    for (let y = startY; y < rect.y + rect.h + GRID_SPACING; y += GRID_SPACING) {
      ctx.moveTo(rect.x, y);
      ctx.lineTo(rect.x + rect.w, y);
    }
    ctx.stroke();

    // World border
    ctx.strokeStyle = 'rgba(255,0,0,0.3)';
    ctx.lineWidth   = 3 / vp.scale;
    ctx.strokeRect(0, 0, State.worldW, State.worldH);
  }

  function drawFood(vp) {
    const food = State.food;
    const rect = vp.getVisibleRect();

    // Group food by color for batching
    const batches = new Array(FOOD_COLORS.length).fill(null).map(() => []);
    for (const [, f] of food) {
      // Cull food outside viewport
      if (f.x < rect.x || f.x > rect.x + rect.w ||
          f.y < rect.y || f.y > rect.y + rect.h) continue;
      const idx = f.colorIdx % FOOD_COLORS.length;
      batches[idx].push(f);
    }

    for (let i = 0; i < batches.length; i++) {
      if (batches[i].length === 0) continue;
      ctx.fillStyle = FOOD_COLORS[i];
      ctx.beginPath();
      for (const f of batches[i]) {
        const r = Math.sqrt(f.mass || 1) * 10;
        ctx.moveTo(f.x + r, f.y);
        ctx.arc(f.x, f.y, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  function drawCells(vp, alpha) {
    const ownIds = State.ownCellIds;
    const interpolated = State.getInterpolated(alpha);

    // Sort: smaller cells on top (render larger first)
    interpolated.sort((a, b) => b.mass - a.mass);

    const minLabelRadiusPx = 20;   // screen pixels
    const ownPlayerId = State.playerId;

    for (const c of interpolated) {
      const r = Math.sqrt(c.mass) * 10;  // radius in world units
      const isOwn = ownIds.has(c.id);
      const color = getPlayerColor(c.playerId);

      // Cell body
      ctx.beginPath();
      ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Border
      ctx.lineWidth   = Math.max(2 / vp.scale, r * 0.05);
      ctx.strokeStyle = isOwn ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.4)';
      ctx.stroke();

      // Labels (name + mass) — only if cell is large enough on screen
      const screenR = r * vp.scale;
      if (screenR > minLabelRadiusPx) {
        const name  = State.getName(c.playerId);
        const label = name || '';
        const mass  = Math.floor(c.mass);

        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';

        // Scale font with cell but cap it
        const fontSize = Math.min(r * 0.4, 40);
        ctx.font        = `bold ${fontSize}px sans-serif`;
        ctx.fillStyle   = 'rgba(255,255,255,0.95)';
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.lineWidth   = fontSize * 0.08;

        if (screenR > 40) {
          ctx.strokeText(label, c.x, c.y - fontSize * 0.4);
          ctx.fillText(label,   c.x, c.y - fontSize * 0.4);

          const massFontSize = fontSize * 0.65;
          ctx.font = `${massFontSize}px sans-serif`;
          ctx.strokeText(mass, c.x, c.y + fontSize * 0.5);
          ctx.fillText(mass,   c.x, c.y + fontSize * 0.5);
        } else {
          ctx.strokeText(label, c.x, c.y);
          ctx.fillText(label,   c.x, c.y);
        }
      }
    }
  }

  function render(vp, alpha) {
    if (!canvas) return;

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Apply viewport transform
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(vp.scale, vp.scale);
    ctx.translate(-vp.x, -vp.y);

    drawGrid(vp);
    drawFood(vp);
    drawCells(vp, alpha);

    ctx.restore();

    // HUD: own mass
    const ownCells = State.getOwnCells();
    if (ownCells.length > 0) {
      const totalMass = ownCells.reduce((s, c) => s + c.mass, 0);
      const hudEl = document.getElementById('hud');
      if (hudEl) hudEl.textContent = `Mass: ${Math.floor(totalMass)}`;
    }
  }

  return { init, render, resizeCanvas };
})();
