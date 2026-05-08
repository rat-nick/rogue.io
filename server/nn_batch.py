"""Batched numpy inference for NEAT recurrent networks.

Each tick: hidden_new = activation(W @ [inputs; hidden_prev] + bias)
Hidden state persists across ticks. On plan rebuild, state is carried over
for surviving bots so recurrent memory is not wiped when another bot dies.
"""
from __future__ import annotations

import numpy as np
import neat

_ACT_TANH    = 0
_ACT_SIGMOID = 1
_ACT_RELU    = 2
_ACT_MAP: dict[str, int] = {'tanh': _ACT_TANH, 'sigmoid': _ACT_SIGMOID, 'relu': _ACT_RELU}


class BatchPlan:
    """
    Weight matrices + mutable hidden state for a fixed set of bots.
    Rebuild via build_batch_plan() whenever the bot set changes.
    """

    def __init__(
        self,
        player_ids:       list[int],
        W:                np.ndarray,   # float32 [B, n_hidden, n_inputs + n_hidden]
        bias:             np.ndarray,   # float32 [B, n_hidden]
        act_ids:          np.ndarray,   # int8    [B, n_hidden]
        output_local_idx: list[int],    # positions in hidden state for output nodes
        state:            np.ndarray,   # float32 [B, n_hidden]  — persistent
    ) -> None:
        self.player_ids        = player_ids
        self._W                = W
        self._bias             = bias
        self._act_ids          = act_ids
        self._output_local_idx = np.array(output_local_idx, dtype=np.int32)
        self._state            = state

    def run(self, inputs_batch: np.ndarray) -> np.ndarray:
        """
        inputs_batch: float32 [B, n_inputs]
        returns:      float32 [B, n_outputs]
        """
        full        = np.concatenate([inputs_batch, self._state], axis=1)
        pre         = np.einsum('bts,bs->bt', self._W, full) + self._bias
        self._state = _apply_activations(pre, self._act_ids)
        return self._state[:, self._output_local_idx]


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
    player_ids: list[int],
    genomes:    list[neat.DefaultGenome],
    neat_cfg:   neat.Config,
    prior_plan: BatchPlan | None = None,
) -> BatchPlan:
    """
    Build a BatchPlan. Pass prior_plan to carry hidden state over for
    bots that existed in the previous plan (avoids memory wipe on bot death).
    """
    gc          = neat_cfg.genome_config
    input_keys  = list(gc.input_keys)
    output_keys = list(gc.output_keys)
    B           = len(player_ids)
    n_inputs    = len(input_keys)

    # Union of all non-input nodes (hidden + output) across all genomes
    all_non_input: set[int] = set(output_keys)
    for genome in genomes:
        for k in genome.nodes:
            if k not in input_keys:
                all_non_input.add(k)
    non_input_keys = sorted(all_non_input)
    n_hidden = len(non_input_keys)
    n_total  = n_inputs + n_hidden

    # Source column index for each node: inputs first, then non-input
    src_col: dict[int, int] = {k: i for i, k in enumerate(input_keys)}
    for i, k in enumerate(non_input_keys):
        src_col[k] = n_inputs + i

    # Row index within hidden state for each non-input node
    hidden_local: dict[int, int] = {k: i for i, k in enumerate(non_input_keys)}

    W       = np.zeros((B, n_hidden, n_total), dtype=np.float32)
    bias    = np.zeros((B, n_hidden),          dtype=np.float32)
    act_ids = np.zeros((B, n_hidden),          dtype=np.int8)

    for b, genome in enumerate(genomes):
        for k in non_input_keys:
            t  = hidden_local[k]
            ng = genome.nodes.get(k)
            if ng is not None:
                bias[b, t]    = ng.bias
                act_ids[b, t] = _ACT_MAP.get(ng.activation, _ACT_TANH)
        for (src, dst), cg in genome.connections.items():
            if not cg.enabled or dst not in hidden_local:
                continue
            s = src_col.get(src)
            if s is None:
                continue
            W[b, hidden_local[dst], s] = cg.weight

    output_local_idx = [hidden_local[k] for k in output_keys if k in hidden_local]

    # Carry over state for bots that survived from the prior plan
    state = np.zeros((B, n_hidden), dtype=np.float32)
    if prior_plan is not None and prior_plan._state.shape[1] > 0:
        prior_pid_idx = {pid: i for i, pid in enumerate(prior_plan.player_ids)}
        n_carry = min(prior_plan._state.shape[1], n_hidden)
        for i, pid in enumerate(player_ids):
            j = prior_pid_idx.get(pid)
            if j is not None:
                state[i, :n_carry] = prior_plan._state[j, :n_carry]

    return BatchPlan(
        player_ids       = player_ids,
        W                = W,
        bias             = bias,
        act_ids          = act_ids,
        output_local_idx = output_local_idx,
        state            = state,
    )
