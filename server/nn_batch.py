"""Batched numpy inference for NEAT recurrent networks.

Each tick: hidden_new = activation(W @ [inputs; hidden_prev] + bias)
Hidden state persists across ticks. On plan rebuild, state is carried over
for surviving bots so recurrent memory is not wiped when another bot dies.

W has shape (B, max_n_hidden, n_inputs + max_n_hidden) where max_n_hidden is
the largest per-genome hidden-node count across all B bots. Each bot uses only
its own local slice W[b, :n_local_b, :n_inputs+n_local_b]; unused rows/cols
remain zero. This avoids the union-of-all-genomes indexing that grows as
B × avg_nodes and can blow out to tens of GB with diverse populations.
"""
from __future__ import annotations

import numpy as np
import neat

_ACT_TANH    = 0
_ACT_SIGMOID = 1
_ACT_RELU    = 2
_ACT_MAP: dict[str, int] = {'tanh': _ACT_TANH, 'sigmoid': _ACT_SIGMOID, 'relu': _ACT_RELU}

# Eligibility trace: credit the last N ticks with exponential decay.
# Effective horizon ≈ 1/(1-decay) ticks. With TICK_RATE=20:
#   decay 0.92, len 40 → ~12.5-tick (~0.6 s) horizon — long enough for
#   chase/intercept/cornering rewards to flow back to the steering
#   activations that initiated the maneuver, without smearing credit
#   across unrelated decisions.
_TRACE_LEN   = 40
_TRACE_DECAY = 0.92


class BatchPlan:
    """
    Weight matrices + mutable hidden state for a fixed set of bots.
    Rebuild via build_batch_plan() whenever the bot set changes.
    """

    def __init__(
        self,
        player_ids:       list[int],
        W:                np.ndarray,   # float32 [B, max_n_hidden, n_inputs + max_n_hidden]
        bias:             np.ndarray,   # float32 [B, max_n_hidden]
        act_ids:          np.ndarray,   # int8    [B, max_n_hidden]
        output_local_idx: np.ndarray,   # int32   [B, n_outputs] — per-bot output positions
        state:            np.ndarray,   # float32 [B, max_n_hidden] — persistent
        conn_map:         list[list[tuple[int, int, int, int]]],  # per-bot: (src_key, dst_key, t, s)
        local_node_keys:  list[list[int]],  # per-bot sorted non-input node keys
        trace_decay:      float = _TRACE_DECAY,
    ) -> None:
        self.player_ids          = player_ids
        self._W                  = W
        self._bias               = bias
        self._act_ids            = act_ids
        self._output_local_idx   = output_local_idx        # (B, n_outputs)
        self._state              = state
        self._conn_map           = conn_map
        self._local_node_keys    = local_node_keys
        self._n_local            = np.array([len(k) for k in local_node_keys], dtype=np.int32)
        self._trace_decay        = trace_decay
        self._last_full: np.ndarray | None = None
        # Circular eligibility trace buffers — allocated lazily on first run()
        self._full_trace:  np.ndarray | None = None  # [TRACE_LEN, B, n_total]
        self._state_trace: np.ndarray | None = None  # [TRACE_LEN, B, max_n_hidden]
        self._trace_ptr:   int = 0
        self._trace_count: int = 0

    def run(self, inputs_batch: np.ndarray) -> np.ndarray:
        """
        inputs_batch: float32 [B, n_inputs]
        returns:      float32 [B, n_outputs]
        """
        full            = np.concatenate([inputs_batch, self._state], axis=1)
        self._last_full = full
        pre             = np.einsum('bts,bs->bt', self._W, full) + self._bias
        self._state     = _apply_activations(pre, self._act_ids)

        # Write (pre, post) pair into the circular eligibility trace
        B, n_total  = full.shape
        _, n_hidden = self._state.shape
        if self._full_trace is None:
            self._full_trace  = np.zeros((_TRACE_LEN, B, n_total),  dtype=np.float32)
            self._state_trace = np.zeros((_TRACE_LEN, B, n_hidden), dtype=np.float32)
        self._full_trace[self._trace_ptr]  = full
        self._state_trace[self._trace_ptr] = self._state
        self._trace_ptr   = (self._trace_ptr + 1) % _TRACE_LEN
        self._trace_count = min(self._trace_count + 1, _TRACE_LEN)

        # Per-bot output gather: _output_local_idx is (B, n_outputs)
        B = self._state.shape[0]
        return self._state[np.arange(B)[:, None], self._output_local_idx]

    def apply_reward(self, bot_idx: int, reward: float, lr: float = 0.004) -> None:
        """Reward-modulated Hebbian update with eligibility trace.

        Applies credit to the last _TRACE_LEN ticks with exponential decay:
            ΔW = lr * reward * Σ_{k=0}^{N-1} λ^k * post_{t-k} ⊗ pre_{t-k}
        where k=0 is the most recent tick. Weights are clipped to [-8, 8].
        Only the bot's own local rows/cols are updated; zero-padded slots stay zero.
        """
        if self._full_trace is None or self._trace_count == 0:
            return
        n_local  = int(self._n_local[bot_idx])
        n_inputs = self._W.shape[2] - self._W.shape[1]   # n_total - max_n_hidden
        n_src    = n_inputs + n_local                     # local input+hidden cols
        dW = np.zeros((n_local, n_src), dtype=np.float32)
        for k in range(self._trace_count):
            idx = (self._trace_ptr - 1 - k) % _TRACE_LEN
            w   = lr * reward * (self._trace_decay ** k)
            dW += w * (
                self._state_trace[idx, bot_idx, :n_local, np.newaxis]
                * self._full_trace[idx, bot_idx, np.newaxis, :n_src]
            )
        W_local = self._W[bot_idx, :n_local, :n_src]
        self._W[bot_idx, :n_local, :n_src] = np.clip(W_local + dW, -8.0, 8.0)

    def write_back_single(self, bot_idx: int, genome: 'neat.DefaultGenome') -> None:
        """Sync Hebbian-modified weights back into the genome's connection genes."""
        for src_key, dst_key, t, s in self._conn_map[bot_idx]:
            cg = genome.connections.get((src_key, dst_key))
            if cg is not None and cg.enabled:
                cg.weight = float(self._W[bot_idx, t, s])


def _apply_activations(pre: np.ndarray, act_ids: np.ndarray) -> np.ndarray:
    out    = np.empty_like(pre)
    tanh_m = act_ids == _ACT_TANH
    sig_m  = act_ids == _ACT_SIGMOID
    relu_m = act_ids == _ACT_RELU
    if tanh_m.any():
        out[tanh_m] = np.tanh(pre[tanh_m])
    if sig_m.any():
        out[sig_m]  = 1.0 / (1.0 + np.exp(-pre[sig_m]))
    if relu_m.any():
        out[relu_m] = np.maximum(0.0, pre[relu_m])
    return out


def build_batch_plan(
    player_ids:  list[int],
    genomes:     list[neat.DefaultGenome],
    neat_cfg:    neat.Config,
    prior_plan:  BatchPlan | None = None,
    trace_decay: float = _TRACE_DECAY,
) -> BatchPlan:
    """
    Build a BatchPlan using per-genome local node re-indexing.

    Each genome gets its own dense local indices 0..n_local_b-1 for its hidden
    nodes. W is sized to (B, max_n_hidden, n_inputs + max_n_hidden) where
    max_n_hidden = max(n_local_b). This keeps W small even when genomes have
    diverse NEAT node IDs, avoiding the union-based O(B * avg_nodes) blowup.
    """
    gc             = neat_cfg.genome_config
    input_keys     = list(gc.input_keys)
    output_keys    = list(gc.output_keys)
    input_key_set  = set(input_keys)
    B              = len(player_ids)
    n_inputs       = len(input_keys)
    n_outputs      = len(output_keys)

    # Per-genome sorted local non-input node key lists
    per_genome_keys: list[list[int]] = []
    for genome in genomes:
        local = sorted(k for k in genome.nodes if k not in input_key_set)
        per_genome_keys.append(local)

    max_n_hidden = max((len(k) for k in per_genome_keys), default=1)
    n_total      = n_inputs + max_n_hidden

    W               = np.zeros((B, max_n_hidden, n_total), dtype=np.float32)
    bias            = np.zeros((B, max_n_hidden),           dtype=np.float32)
    act_ids         = np.zeros((B, max_n_hidden),           dtype=np.int8)
    output_local_idx = np.zeros((B, n_outputs),             dtype=np.int32)
    conn_map: list[list[tuple[int, int, int, int]]] = []

    for b, (genome, local_keys) in enumerate(zip(genomes, per_genome_keys)):
        hidden_local: dict[int, int] = {k: i for i, k in enumerate(local_keys)}
        src_col: dict[int, int] = {k: i for i, k in enumerate(input_keys)}
        for i, k in enumerate(local_keys):
            src_col[k] = n_inputs + i

        for k in local_keys:
            t  = hidden_local[k]
            ng = genome.nodes.get(k)
            if ng is not None:
                bias[b, t]    = ng.bias
                act_ids[b, t] = _ACT_MAP.get(ng.activation, _ACT_TANH)

        bot_conns: list[tuple[int, int, int, int]] = []
        for (src, dst), cg in genome.connections.items():
            if not cg.enabled or dst not in hidden_local:
                continue
            s = src_col.get(src)
            if s is None:
                continue
            t = hidden_local[dst]
            W[b, t, s] = cg.weight
            bot_conns.append((src, dst, t, s))
        conn_map.append(bot_conns)

        for j, k in enumerate(output_keys):
            if k in hidden_local:
                output_local_idx[b, j] = hidden_local[k]

    # Carry over hidden state (and W when topology is unchanged) for survivors.
    state = np.zeros((B, max_n_hidden), dtype=np.float32)
    if prior_plan is not None and prior_plan._state.shape[1] > 0:
        prior_pid_idx = {pid: i for i, pid in enumerate(prior_plan.player_ids)}
        same_shape    = (prior_plan._W.shape[1] == max_n_hidden and
                         prior_plan._W.shape[2] == n_total)
        for i, pid in enumerate(player_ids):
            j = prior_pid_idx.get(pid)
            if j is None:
                continue
            prior_keys = prior_plan._local_node_keys[j]
            curr_keys  = per_genome_keys[i]
            if prior_keys == curr_keys:
                # Topology unchanged: copy state and (if shape matches) W directly.
                n_carry = min(len(prior_keys), prior_plan._state.shape[1], max_n_hidden)
                state[i, :n_carry] = prior_plan._state[j, :n_carry]
                if same_shape:
                    W[i] = prior_plan._W[j]
            else:
                # Topology changed: map surviving nodes by key.
                prior_hidden = {k: idx for idx, k in enumerate(prior_keys)}
                n_prior_state = prior_plan._state.shape[1]
                for curr_idx, k in enumerate(curr_keys):
                    prior_idx = prior_hidden.get(k)
                    if prior_idx is not None and prior_idx < n_prior_state:
                        state[i, curr_idx] = prior_plan._state[j, prior_idx]

    return BatchPlan(
        player_ids       = player_ids,
        W                = W,
        bias             = bias,
        act_ids          = act_ids,
        output_local_idx = output_local_idx,
        state            = state,
        conn_map         = conn_map,
        local_node_keys  = per_genome_keys,
        trace_decay      = trace_decay,
    )
