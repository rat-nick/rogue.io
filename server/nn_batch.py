"""Batched numpy inference for NEAT FeedForwardNetworks.

Replaces per-bot net.activate() with a single vectorized forward pass over
all active bots. Uses a superset topology (union of all bots' connections)
with zero-padding for missing connections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import neat
from neat.graphs import feed_forward_layers

_ACT_TANH    = 0
_ACT_SIGMOID = 1
_ACT_RELU    = 2
_ACT_MAP: dict[str, int] = {'tanh': _ACT_TANH, 'sigmoid': _ACT_SIGMOID, 'relu': _ACT_RELU}


@dataclass
class _LayerPlan:
    W:       np.ndarray  # float32 [B, tgt_count, src_count]
    bias:    np.ndarray  # float32 [B, tgt_count]
    act_ids: np.ndarray  # int8    [B, tgt_count]  0=tanh 1=sigmoid 2=relu
    tgt_idx: list[int]   # global buffer indices for target nodes
    src_idx: list[int]   # global buffer indices for source nodes


class BatchPlan:
    """
    Immutable inference plan for a fixed set of bots.
    Rebuild via build_batch_plan() whenever bots are added or removed.
    """

    def __init__(
        self,
        player_ids:  list[int],
        input_idx:   list[int],
        output_idx:  list[int],
        buffer_size: int,
        layers:      list[_LayerPlan],
    ) -> None:
        self.player_ids   = player_ids
        self._input_idx   = np.array(input_idx,  dtype=np.int32)
        self._output_idx  = np.array(output_idx, dtype=np.int32)
        self._buffer_size = buffer_size
        self._layers      = layers

    def run(self, inputs_batch: np.ndarray) -> np.ndarray:
        """
        inputs_batch: float32 [B, 97]  — row i corresponds to player_ids[i]
        returns:      float32 [B,  4]  — network outputs for each bot
        """
        B   = len(self.player_ids)
        buf = np.zeros((B, self._buffer_size), dtype=np.float32)
        buf[:, self._input_idx] = inputs_batch

        for lp in self._layers:
            src = buf[:, lp.src_idx]
            pre = np.einsum('bts,bs->bt', lp.W, src) + lp.bias
            buf[:, lp.tgt_idx] = _apply_activations(pre, lp.act_ids)

        return buf[:, self._output_idx]


def _apply_activations(pre: np.ndarray, act_ids: np.ndarray) -> np.ndarray:
    """Apply per-(bot, node) activation function. Both inputs shape [B, N]."""
    out = np.empty_like(pre)
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
) -> BatchPlan:
    """
    Build a BatchPlan from parallel lists of player_ids and genomes.
    Call this once whenever the active bot set changes.
    """
    gc          = neat_cfg.genome_config
    input_keys  = list(gc.input_keys)
    output_keys = list(gc.output_keys)
    B           = len(player_ids)

    # Union of all enabled connections across all bots
    union_conns: set[tuple[int, int]] = set()
    for genome in genomes:
        for (src, dst), cg in genome.connections.items():
            if cg.enabled:
                union_conns.add((src, dst))

    # Topological layers on the union graph (neat 2.0.0 returns (layers, required))
    layers_sets, _ = feed_forward_layers(input_keys, output_keys, list(union_conns))

    # Assign buffer column indices: inputs first, then non-input nodes in layer order
    node_to_idx: dict[int, int] = {k: i for i, k in enumerate(input_keys)}
    offset = len(input_keys)
    for layer_set in layers_sets:
        for k in sorted(layer_set):
            node_to_idx[k] = offset
            offset += 1

    buffer_size = offset
    input_idx   = [node_to_idx[k] for k in input_keys]
    output_idx  = [node_to_idx[k] for k in output_keys]

    layer_plans: list[_LayerPlan] = []
    processed = set(input_keys)

    for layer_set in layers_sets:
        tgt_keys = sorted(layer_set)

        src_keys_set: set[int] = set()
        for tk in tgt_keys:
            for sk in processed:
                if (sk, tk) in union_conns:
                    src_keys_set.add(sk)
        src_keys    = sorted(src_keys_set, key=lambda k: node_to_idx[k])
        sk_to_local = {sk: i for i, sk in enumerate(src_keys)}

        tgt_count = len(tgt_keys)
        src_count = len(src_keys)

        W       = np.zeros((B, tgt_count, src_count), dtype=np.float32)
        bias    = np.zeros((B, tgt_count),            dtype=np.float32)
        act_ids = np.zeros((B, tgt_count),            dtype=np.int8)

        for b, genome in enumerate(genomes):
            for t, tk in enumerate(tgt_keys):
                node_gene = genome.nodes.get(tk)
                if node_gene is not None:
                    bias[b, t]    = node_gene.bias
                    act_ids[b, t] = _ACT_MAP.get(node_gene.activation, _ACT_TANH)
                for sk in src_keys:
                    cg = genome.connections.get((sk, tk))
                    if cg is not None and cg.enabled:
                        W[b, t, sk_to_local[sk]] = cg.weight

        layer_plans.append(_LayerPlan(
            W       = W,
            bias    = bias,
            act_ids = act_ids,
            tgt_idx = [node_to_idx[k] for k in tgt_keys],
            src_idx = [node_to_idx[k] for k in src_keys],
        ))
        processed.update(layer_set)

    return BatchPlan(
        player_ids  = player_ids,
        input_idx   = input_idx,
        output_idx  = output_idx,
        buffer_size = buffer_size,
        layers      = layer_plans,
    )
