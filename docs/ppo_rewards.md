# PPO Reward Structure

All per-step rewards are scaled by `dt_scale = dt / TICK_INTERVAL` so the signal is consistent regardless of tick rate.

---

## Per-Step Rewards (obs loop — attributed to the *previous* action)

These are computed each tick by comparing the current world state against last tick's state. Because they are measured *after* physics resolves, they reflect the outcome of the action taken one step earlier.

| Condition | Reward | Notes |
|-----------|--------|-------|
| **Center proximity bonus** | `+0.1 × center_prox × dt_scale` | `center_prox ∈ [0, 1]`; 1 = at world center, 0 = at corner. Encourages staying near the middle. |
| **Cell eaten** (`mass_delta ≥ 15`) | `+log(1 + mass_delta) × dt_scale` | Large reward proportional to mass gained by consuming another player's cell. |
| **Food eaten** (`0 < mass_delta < 5`) | `+log(1 + mass_delta) × dt_scale` | Small reward for grazing food pellets. |
| **Mass lost** (`mass_delta ≤ −10`) | `−log(1 + |mass_delta|) × dt_scale` | Penalty for losing significant mass (being eaten, decaying). |
| **Survival bonus** | `+0.01 × dt_scale` | Tiny reward for staying alive each tick. |
| **Death** | `−2.0` (not dt_scaled) | Applied on the tick the bot dies; `done = 1` is set for GAE bootstrap. |

> **Credit assignment note:** `mass_delta` measures change since the *previous* tick, so it always lags the action that caused it by one step. Use the action-loop penalties below for immediate action-level feedback.

---

## Deferred Split Penalty (obs loop — fires `SPLIT_GAIN_WINDOW` ticks after a split)

| Condition | Reward | Notes |
|-----------|--------|-------|
| **Unproductive split** | `−SPLIT_NO_GAIN_PENALTY × dt_scale` | If `total_mass ≤ mass_at_split_time` after `SPLIT_GAIN_WINDOW = 10` ticks, the split is considered wasted. Only one check is pending at a time; further splits while a check is in-flight are ignored. |
| **Productive split** | `0` | No penalty if mass increased during the window. |

Constants: `SPLIT_GAIN_WINDOW = 10`, `SPLIT_NO_GAIN_PENALTY = 1.5`.

---

## Per-Action Penalties (action loop — attributed to the *current* action)

These are computed in the same step the action is stored in the rollout buffer, giving PPO correct credit assignment.

| Action | Reward | Notes |
|--------|--------|-------|
| **Eject** | `−log(1 + n_ejecting × EJECT_MASS_COST) × dt_scale` | Applied every tick the bot fires an eject. `n_ejecting` is the number of cells currently eligible to eject (mass ≥ `EJECT_MIN_MASS = 32`). Mirrors the eating reward scale — ejecting 1 cell costs `~−2.77`, 2 cells `~−3.43`. This gives correct immediate credit assignment, unlike the lagged mass-delta signal. |

---

## Fitness Score (logging only, not used in PPO updates)

The fitness metric is computed on bot death or rollout end for leaderboard display. It does **not** feed into the PPO gradient.

$$
\text{fitness} = 0.4 \times \text{peak\_mass} + 0.3 \times \text{avg\_mass} + 3 \times \text{food\_eaten} + 9 \times \text{cells\_eaten\_mass} + 3 \times \ln(1 + t) - 1000 \times (f_{\text{edge}}^2 + f_{\text{corner}}^2)
$$

Where $t$ is time alive in seconds, $f_{\text{edge}}$ and $f_{\text{corner}}$ are fractions of ticks spent near walls/corners.

---

## Summary

| Signal | When | dt_scaled | Magnitude |
|--------|------|-----------|-----------|
| Center proximity | Every tick | Yes | `0–0.1` |
| Food eaten | When `0 < Δmass < 5` | Yes | `~0.01–0.16` |
| Cell eaten | When `Δmass ≥ 15` | Yes | `~2.8+` |
| Mass lost | When `Δmass ≤ −10` | Yes | `~2.4+` |
| Survival | Every tick | Yes | `0.01` |
| Death | On death | No | `−2.0` |
| Unproductive split | 10 ticks after split | Yes | `−1.5` |
| Eject | Every eject tick | Yes | `−0.3` |
