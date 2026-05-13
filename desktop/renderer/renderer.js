const Renderer = (() => {
  let canvas = null;
  let ctx    = null;

  // Food color palette matching server config.FOOD_COLORS
  const FOOD_COLORS = [
    '#ff5555', '#ffa500', '#ffff55', '#55ff55',
    '#55ffff', '#5555ff', '#ff55ff', '#ffc8c8',
    '#c8ffc8', '#c8c8ff', '#ffffc8', '#c8e6ff',
  ];

  // Map player_id -> HSL color string (cached; cleared when hue data arrives)
  const playerColors = new Map();
  const playerHues   = new Map(); // track which hue was used to build the cached color
  function getPlayerColor(playerId) {
    // Use genome-derived hue if available, otherwise golden-angle fallback
    const hue = State.getHue(playerId) >= 0
      ? State.getHue(playerId)
      : (playerId * 137.508) % 360;
    // Invalidate cache if hue changed (e.g. hue arrived after first render)
    if (playerHues.get(playerId) !== hue) {
      playerHues.set(playerId, hue);
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

  function drawViruses(vp) {
    const viruses = State.viruses;
    const rect = vp.getVisibleRect();

    ctx.fillStyle = '#33ff33';
    ctx.strokeStyle = '#00cc00';
    ctx.lineWidth = 3 / vp.scale;

    for (const [, v] of viruses) {
      // Cull viruses outside viewport
      if (v.x < rect.x - 150 || v.x > rect.x + rect.w + 150 ||
          v.y < rect.y - 150 || v.y > rect.y + rect.h + 150) continue;

      // Draw spiky virus shape
      const r = 100; // Fixed radius for viruses
      const spikes = 12;
      ctx.beginPath();
      for (let i = 0; i < spikes; i++) {
        const angle = (i / spikes) * Math.PI * 2;
        const nextAngle = ((i + 1) / spikes) * Math.PI * 2;
        const midAngle = (angle + nextAngle) / 2;
        
        // Outer spike
        const outerX = v.x + Math.cos(midAngle) * r * 1.3;
        const outerY = v.y + Math.sin(midAngle) * r * 1.3;
        
        if (i === 0) {
          ctx.moveTo(v.x + Math.cos(angle) * r, v.y + Math.sin(angle) * r);
        }
        ctx.lineTo(outerX, outerY);
        ctx.lineTo(v.x + Math.cos(nextAngle) * r, v.y + Math.sin(nextAngle) * r);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
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

    // Second pass: draw species highlight rings on top of all cells
    const selSpecies = window.selectedSpecies;
    const spMap = window.playerSpeciesMap;
    if (selSpecies !== null && selSpecies !== undefined && spMap) {
      ctx.setLineDash([16 / vp.scale, 8 / vp.scale]);
      ctx.strokeStyle = 'rgba(255,220,0,0.9)';
      ctx.lineWidth   = 4 / vp.scale;
      for (const c of interpolated) {
        if (spMap.get(c.playerId) !== selSpecies) continue;
        const r = Math.sqrt(c.mass) * 10;
        ctx.beginPath();
        ctx.arc(c.x, c.y, r + 10 / vp.scale, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.setLineDash([]);
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
    drawViruses(vp);
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
