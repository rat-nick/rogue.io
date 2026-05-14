# PPO Network Observations

Total input size: **122 floats** (`N_OBS = 122`), all normalized to approximately `[0, 1]` or `[-1, 1]`.

The observation vector is built by `_build_inputs_batch_v2` in `server/bot.py`.

---

## Scan Radius

All ray and quadrant features are computed within a **scan radius** that scales with the bot's total mass, mirroring the player viewport:

```
scan_radius = min(VIEW_BASE_SIZE / 2 × (max(mass, 20) / 100) ^ VIEW_MASS_SCALE, 3500)
```

Larger bots see further.

---

## 1. Interoceptive — indices 0–4 (5 values)

Self-information about the bot's own body.

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 0 | `avg_mass_n` | `log(1 + total_mass / n_cells)` normalized by `log(1 + 5000)` | `[0, 1]` |
| 1 | `num_cells_n` | Number of cells / max cells (`_MAX_CELLS`) | `[0, 1]` |
| 2 | `split_ready` | 1 if any cell is heavy enough to split, else 0 | `{0, 1}` |
| 3 | `eject_ready` | 1 if any cell is heavy enough to eject, else 0 | `{0, 1}` |
| 4 | `merge_timer_n` | Longest active merge cooldown / max possible merge timer | `[0, 1]` |

---

## 2. Ray Channels — indices 5–60 (56 values, 8 rays × 7 channels)

8 rays cast uniformly around the bot at 45° increments (ray 0 = East, increasing counter-clockwise). Each ray reports the **closest detected object** of each type along that direction.

Proximity encoding: `1.0 - distance / scan_radius` (0 = at scan boundary, 1 = at bot centroid).

For each ray `r` (r = 0…7), the 7 channels at index `5 + r*7 + c` are:

| Channel offset | Name | Description | Range |
|----------------|------|-------------|-------|
| 0 | `ray_food` | Proximity of nearest food pellet | `[0, 1]` |
| 1 | `ray_prey` | Proximity of nearest edible enemy cell | `[0, 1]` |
| 2 | `ray_prey_mass` | Mass of nearest prey / own mass (relative size) | `[0, ~0.8]` |
| 3 | `ray_threat` | Proximity of nearest threatening enemy cell | `[0, 1]` |
| 4 | `ray_threat_mass` | own mass / threat mass (1 = borderline threat, ~0 = huge threat) | `[0, 1]` |
| 5 | `ray_virus` | Proximity of nearest virus | `[0, 1]` |
| 6 | `rw` | Wall proximity along this ray direction | `[0, 1]` |

**Prey** = enemy cell with mass ≤ `largest_own_cell / EAT_RATIO` (can be eaten).  
**Threat** = enemy cell with mass ≥ `largest_own_cell × EAT_RATIO` (can eat us).

---

## 3. Quadrant Summaries — indices 61–68 (8 values)

The scan area is divided into 4 quadrants (NE, NW, SW, SE relative to bot centroid). These give a density overview rather than directional precision.

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 61 | `qf_0` | Food density in SE quadrant — `count / (π·FOOD_TARGET·R²/(4·W·H))`, 1.0 = global-average density | `[0, 1]` |
| 62 | `qf_1` | Food density in SW quadrant (same normalisation) | `[0, 1]` |
| 63 | `qf_2` | Food density in NW quadrant (same normalisation) | `[0, 1]` |
| 64 | `qf_3` | Food density in NE quadrant (same normalisation) | `[0, 1]` |
| 65 | `qt_0` | Total threat mass in SE quadrant / 5000 | `[0, 1]` |
| 66 | `qt_1` | Total threat mass in SW quadrant / 5000 | `[0, 1]` |
| 67 | `qt_2` | Total threat mass in NW quadrant / 5000 | `[0, 1]` |
| 68 | `qt_3` | Total threat mass in NE quadrant / 5000 | `[0, 1]` |

---

## 4. Contextual — indices 69–71 (3 values)

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 69 | `center_dist` | Distance from world center / half-diagonal — 0 = center, 1 = corner | `[0, 1]` |
| 70 | `mass_rank` | Fraction of visible enemies that are prey (0 = all threats, 1 = all prey) | `[0, 1]` |
| 71 | `danger_score` | Weighted sum of threat proximity × relative mass, normalized by 10 | `[0, 1]` |

---

## 5. Velocity — indices 72–73 (2 values)

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 72 | `vel_x` | Centroid displacement X since last tick / `_VEL_NORM` | `[-1, 1]` |
| 73 | `vel_y` | Centroid displacement Y since last tick / `_VEL_NORM` | `[-1, 1]` |

---

## 6. Per-Cell Body Features — indices 74–121 (48 values, 16 cells × 3)

The bot's own cells, sorted by mass **descending** (slot 0 = heaviest cell). Slots for cells that don't exist are zero-padded.

For each cell slot `k` (k = 0…15), the 3 values at index `74 + k*3` are:

| Offset | Name | Description | Range |
|--------|------|-------------|-------|
| 0 | `dx_n` | `(cell.x - centroid.x) / scan_radius`, clamped | `[-1, 1]` |
| 1 | `dy_n` | `(cell.y - centroid.y) / scan_radius`, clamped | `[-1, 1]` |
| 2 | `mass_n` | `log(1 + cell.mass) / log(1 + 5000)` | `[0, 1]` |

---

## Summary

| Group | Indices | Count |
|-------|---------|-------|
| Interoceptive | 0–4 | 5 |
| Ray channels (8 rays × 7) | 5–60 | 56 |
| Quadrant food (4) | 61–64 | 4 |
| Quadrant threat (4) | 65–68 | 4 |
| Contextual | 69–71 | 3 |
| Velocity | 72–73 | 2 |
| Per-cell body (16 cells × 3) | 74–121 | 48 |
| **Total** | | **122** |
