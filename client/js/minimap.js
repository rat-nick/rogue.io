const Minimap = (() => {
  let canvas = null;
  let ctx    = null;
  const SIZE = 200;

  function init(c) {
    canvas = c;
    ctx    = c.getContext('2d');
    canvas.width  = SIZE;
    canvas.height = SIZE;
  }

  function render(vp) {
    if (!canvas || !State.ready) return;

    const ww = State.worldW;
    const wh = State.worldH;
    const sx = SIZE / ww;
    const sy = SIZE / wh;

    // Background
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(0, 0, SIZE, SIZE);

    // Food (single-pixel dots, batch all at once)
    ctx.fillStyle = 'rgba(180,180,180,0.5)';
    for (const [, f] of State.food) {
      ctx.fillRect(f.x * sx, f.y * sy, 1, 1);
    }

    // Cells
    const ownIds = State.ownCellIds;
    for (const [, c] of State.cells) {
      const r = Math.max(Math.sqrt(c.mass) * 10 * sx, 2);
      const mx = c.x * sx;
      const my = c.y * sy;
      ctx.beginPath();
      ctx.arc(mx, my, r, 0, Math.PI * 2);
      if (ownIds.has(c.id)) {
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
      } else {
        const hue = (c.playerId * 137.508) % 360;
        ctx.fillStyle = `hsla(${hue}, 70%, 55%, 0.7)`;
      }
      ctx.fill();
    }

    // Viewport rectangle
    const rect = vp.getVisibleRect();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth   = 1;
    ctx.strokeRect(
      rect.x * sx,
      rect.y * sy,
      rect.w * sx,
      rect.h * sy
    );
  }

  return { init, render };
})();
