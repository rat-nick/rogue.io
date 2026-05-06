const Viewport = (() => {
  const VIEW_BASE_SIZE = 3840;
  const VIEW_MASS_SCALE = 0.4;
  const LERP_SPEED = 0.12; // per frame smoothing factor

  let canvas = null;
  let x = 0;
  let y = 0; // camera center in world coords
  let scale = 1; // pixels per world unit
  let targetX = 0;
  let targetY = 0;
  let targetScale = 1;

  function init(c) {
    canvas = c;
    x = targetX = 10000;
    y = targetY = 10000;
    scale = targetScale = canvas.width / VIEW_BASE_SIZE;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function update(ownCells, dt) {
    if (!canvas || ownCells.length === 0) return;

    // Compute centroid (mass-weighted)
    let totalMass = 0;
    let cx = 0;
    let cy = 0;
    for (const c of ownCells) {
      cx += c.x * c.mass;
      cy += c.y * c.mass;
      totalMass += c.mass;
    }
    if (totalMass > 0) {
      cx /= totalMass;
      cy /= totalMass;
    }

    targetX = cx;
    targetY = cy;

    // Compute target scale
    const vw = VIEW_BASE_SIZE * Math.pow(Math.max(totalMass, 1) / 100, VIEW_MASS_SCALE);
    targetScale = canvas.width / vw;

    // Frame-rate independent lerp: equivalent to LERP_SPEED per 16.67ms frame
    const t = 1 - Math.pow(1 - LERP_SPEED, (dt || 16.67) / 16.67);
    x = lerp(x, targetX, t);
    y = lerp(y, targetY, t);
    scale = lerp(scale, targetScale, t);
  }

  function worldToScreen(wx, wy) {
    return {
      sx: (wx - x) * scale + canvas.width / 2,
      sy: (wy - y) * scale + canvas.height / 2,
    };
  }

  function screenToWorld(sx, sy) {
    return {
      wx: (sx - canvas.width / 2) / scale + x,
      wy: (sy - canvas.height / 2) / scale + y,
    };
  }

  function getVisibleRect() {
    const hw = canvas.width / 2 / scale;
    const hh = canvas.height / 2 / scale;
    return { x: x - hw, y: y - hh, w: hw * 2, h: hh * 2 };
  }

  return {
    get x() {
      return x;
    },
    get y() {
      return y;
    },
    get scale() {
      return scale;
    },
    init,
    update,
    worldToScreen,
    screenToWorld,
    getVisibleRect,
  };
})();
