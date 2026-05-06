const State = (() => {
  // World constants (set on init)
  let worldW = 20000;
  let worldH = 20000;
  let tickRate = 20;
  let playerId = null;
  let ready = false;

  // Live data
  const cells = new Map(); // id -> {id, x, y, mass, playerId, name, prevX, prevY, prevMass}
  const food = new Map(); // id -> {id, x, y, colorIdx}
  const ownCellIds = new Set();
  let leaderboard = [];
  let lastTickTime = 0;
  let smoothTickMs = 50; // EMA of actual inter-tick arrival interval
  let lastTickNum = -1;

  // Name cache: playerId -> name string
  const nameCache = new Map();

  function init(msg) {
    playerId = msg[1];
    worldW = msg[2];
    worldH = msg[3];
    tickRate = msg[4];
    ready = true;
    cells.clear();
    food.clear();
    ownCellIds.clear();
    nameCache.clear();
    leaderboard = [];
  }

  function applyTick(msg) {
    // msg layout: [0x11, tick_num, own_cell_ids, visible_cells, food_new, food_removed, leaderboard_or_null]
    const tickNum = msg[1];
    const ownIds = msg[2]; // array of cell IDs owned by this player
    const visCells = msg[3]; // [[id, x, y, mass, player_id, name_or_null], ...]
    const foodNew = msg[4]; // [[id, x, y, color_idx], ...]
    const foodRemoved = msg[5]; // [id, ...]
    const lb = msg[6]; // [[name, score], ...] or null

    lastTickNum = tickNum;
    const now = performance.now();
    if (lastTickTime > 0) {
      const interval = now - lastTickTime;
      // Sanity-clamp: ignore outliers (paused tab, etc.)
      if (interval > 10 && interval < 300) {
        smoothTickMs = smoothTickMs * 0.9 + interval * 0.1;
      }
    }
    lastTickTime = now;

    // Update own cell set
    ownCellIds.clear();
    for (const id of ownIds) ownCellIds.add(id);

    // Update visible cells (save previous positions for interpolation)
    const seenIds = new Set();
    for (const c of visCells) {
      const [id, x, y, mass, pid, name] = c;
      seenIds.add(id);
      if (name !== null && name !== undefined) nameCache.set(pid, name);
      const existing = cells.get(id);
      if (existing) {
        existing.prevX = existing.x;
        existing.prevY = existing.y;
        existing.prevMass = existing.mass;
        existing.x = x;
        existing.y = y;
        existing.mass = mass;
        existing.playerId = pid;
      } else {
        cells.set(id, {
          id,
          x,
          y,
          mass,
          playerId: pid,
          prevX: x,
          prevY: y,
          prevMass: mass,
        });
      }
    }

    // Remove cells no longer visible
    for (const [id] of cells) {
      if (!seenIds.has(id)) cells.delete(id);
    }

    // Delta food
    for (const f of foodNew) food.set(f[0], { id: f[0], x: f[1], y: f[2], colorIdx: f[3], mass: f[4] });
    for (const id of foodRemoved) food.delete(id);

    // Leaderboard (sent every N ticks)
    if (lb) leaderboard = lb;
  }

  function getOwnCells() {
    const result = [];
    for (const id of ownCellIds) {
      const c = cells.get(id);
      if (c) result.push(c);
    }
    return result;
  }

  function getName(pid) {
    return nameCache.get(pid) || '?';
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function getInterpolated(alpha) {
    // Returns array of cells with interpolated positions
    const result = [];
    const a = Math.min(alpha, 1.0);
    for (const [, c] of cells) {
      result.push({
        id: c.id,
        x: lerp(c.prevX, c.x, a),
        y: lerp(c.prevY, c.y, a),
        mass: lerp(c.prevMass, c.mass, a),
        playerId: c.playerId,
      });
    }
    return result;
  }

  return {
    get ready() {
      return ready;
    },
    get playerId() {
      return playerId;
    },
    get worldW() {
      return worldW;
    },
    get worldH() {
      return worldH;
    },
    get tickRate() {
      return tickRate;
    },
    get cells() {
      return cells;
    },
    get food() {
      return food;
    },
    get ownCellIds() {
      return ownCellIds;
    },
    get leaderboard() {
      return leaderboard;
    },
    get lastTickTime() {
      return lastTickTime;
    },
    get smoothTickMs() {
      return smoothTickMs;
    },
    getOwnCells,
    getName,
    getInterpolated,
    init,
    applyTick,
    reset() {
      ready = false;
      playerId = null;
      cells.clear();
      food.clear();
      ownCellIds.clear();
      leaderboard = [];
      nameCache.clear();
    },
  };
})();
