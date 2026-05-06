/**
 * SpectateViewport — drop-in replacement for Viewport used by the spectator client.
 * Supports smooth following of a target player and an independent zoom multiplier.
 *
 * Zoom convention (matches ZOOM_MULTIPLIERS in server/protocol.py):
 *   higher multiplier  → more zoomed IN  (smaller visible world area)
 *   lower  multiplier  → more zoomed OUT (larger visible world area)
 */
const SpectateViewport = (() => {
  const LERP_SPEED   = 0.10;   // per-frame smoothing (matches Viewport)
  const VIEW_BASE    = 3840;   // world units visible at base mass (matches config.VIEW_BASE_SIZE)
  const VIEW_MASS_EX = 0.4;    // exponent (matches config.VIEW_MASS_SCALE)

  let canvas       = null;
  let x            = 15000;
  let y            = 15000;
  let scale        = 1;
  let targetX      = 15000;
  let targetY      = 15000;
  let targetScale  = 1;
  let zoomMult     = 1.0;   // zoom multiplier; 1.0 = normal follow zoom

  // Last known position from stats (used as fallback while cells load)
  let fallbackX    = null;
  let fallbackY    = null;
  let fallbackMass = null;

  function lerp(a, b, t) { return a + (b - a) * t; }

  function init(c) {
    canvas = c;
    const ww = (typeof State !== 'undefined' && State.ready) ? State.worldW : 30000;
    const wh = (typeof State !== 'undefined' && State.ready) ? State.worldH : 30000;
    x = targetX = ww / 2;
    y = targetY = wh / 2;
    // Default: fit whole world
    scale = targetScale = canvas.width / (ww * 1.05);
  }

  /**
   * @param {Array}  followCells  Cells belonging to the followed player (may be empty).
   * @param {number} dt           Frame delta in ms.
   * @param {Array|null} fallback [id, name, mass, cx, cy, cells, isBot] from stats — used
   *                              when followCells is empty but we still want to pan there.
   */
  function update(followCells, dt, fallback) {
    if (!canvas) return;

    const t = 1 - Math.pow(1 - LERP_SPEED, (dt || 16.67) / 16.67);

    if (followCells && followCells.length > 0) {
      // Mass-weighted centroid of followed player's cells
      let totalMass = 0, cx = 0, cy = 0;
      for (const c of followCells) {
        cx        += c.x * c.mass;
        cy        += c.y * c.mass;
        totalMass += c.mass;
      }
      if (totalMass > 0) { cx /= totalMass; cy /= totalMass; }

      targetX = cx;
      targetY = cy;
      // Scale: same formula as server-side viewport_rect, then divided by zoom mult
      const vw = VIEW_BASE * Math.pow(Math.max(totalMass, 1) / 100, VIEW_MASS_EX) / zoomMult;
      targetScale = canvas.width / vw;

    } else if (fallback) {
      // Cells not yet in viewport — pan toward last-known stats position
      targetX = fallback[3];  // cx from stats
      targetY = fallback[4];  // cy from stats
      const mass = Math.max(fallback[2] || 1, 1);
      const vw   = VIEW_BASE * Math.pow(mass / 100, VIEW_MASS_EX) / zoomMult;
      targetScale = canvas.width / vw;

    } else {
      // Overview: fit entire world
      const ww = (typeof State !== 'undefined' && State.ready) ? State.worldW : 30000;
      const wh = (typeof State !== 'undefined' && State.ready) ? State.worldH : 30000;
      targetX = ww / 2;
      targetY = wh / 2;
      // zoom in/out relative to full-world view
      targetScale = (canvas.width / (ww * 1.05)) * zoomMult;
    }

    x     = lerp(x,     targetX,     t);
    y     = lerp(y,     targetY,     t);
    scale = lerp(scale, targetScale, t);
  }

  function setZoomMultiplier(m) { zoomMult = m; }

  function worldToScreen(wx, wy) {
    return {
      sx: (wx - x) * scale + canvas.width  / 2,
      sy: (wy - y) * scale + canvas.height / 2,
    };
  }

  function screenToWorld(sx, sy) {
    return {
      wx: (sx - canvas.width  / 2) / scale + x,
      wy: (sy - canvas.height / 2) / scale + y,
    };
  }

  function getVisibleRect() {
    const hw = canvas.width  / 2 / scale;
    const hh = canvas.height / 2 / scale;
    return { x: x - hw, y: y - hh, w: hw * 2, h: hh * 2 };
  }

  return {
    get x()     { return x;     },
    get y()     { return y;     },
    get scale() { return scale; },
    init,
    update,
    setZoomMultiplier,
    worldToScreen,
    screenToWorld,
    getVisibleRect,
  };
})();
