"""Positive listening: does the receiver actually use the learned symbols?

`analyze_channel_mi.py` established the emitters do encode state -- I(m ; s | sender)
sits ~100x above the estimator's noise floor on every non-collapsed arm. That kills
the "the RL learns no protocol" reading and moves the question downstream: the code
exists, but is anyone reading it?

Same weights, same held-out seed, same episodes; only what travels on the wire
changes:

    learned    the symbols the network actually chose (argmax)
    random     uniform random symbols        -> destroys the content, keeps the format
    constant   every slot forced to symbol 0 -> destroys it a different way

A node can always tell that a *direct* neighbour has gone silent: that is the physical
channel (the -1 sentinel of loss/death), not something the message carries, and it
stays available in all three conditions. So the random/constant rows measure what
silence alone buys, and the gap up to `learned` is what the learned code is worth to
the receiver. A small gap means the receiver ignores a channel that demonstrably
carries information -- a listening failure, not a signalling one.
"""
import argparse
import json
import os

import numpy as np
import torch

from env import Environment
from grid_adjacency import build_grid_adjacency
from iql import IqlAlgorithms
from network import Network


def build(ckpt_path, seed, failure_rate=0.01, p_loss_max=0.05):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    grid = ckpt["grid"]
    adjacency = build_grid_adjacency(grid, grid)
    n = len(adjacency)

    nw_config = dict(ckpt["nw_config"])
    network = Network(**nw_config, seed=ckpt["seed"])
    dropped = network.load_checkpoint_state(ckpt["state_dict"])
    if dropped:
        print(f"[info] dropped {dropped} legacy trust-gate tensors "
              "(never read by forward; see Network.load_checkpoint_state)")
    network.eval()

    env = Environment(
        adjacency_matrix=adjacency, failure_rate=failure_rate,
        length_message=nw_config["length_message"],
        message_candidates=nw_config["message_candidates"],
        p_loss_max=p_loss_max, rng=np.random.default_rng(seed=seed),
    )
    iql = IqlAlgorithms(env, network, rng=np.random.default_rng(seed=seed))
    return iql, network, n, ckpt, len(nw_config["message_candidates"])


def patch_wire(iql, mode, n_candidates, rng):
    """Wrap get_actions_batch so only the emitted Q-symbols are rewritten."""
    original = iql.network.get_actions_batch

    def wrapped(frames, hidden, node_ids):
        belief_probs, q_ids, h_out = original(frames, hidden, node_ids)
        if q_ids.size:
            if mode == "random":
                q_ids = rng.integers(0, n_candidates, size=q_ids.shape).astype(q_ids.dtype)
            elif mode == "constant":
                q_ids = np.zeros_like(q_ids)
        return belief_probs, q_ids, h_out

    iql.network.get_actions_batch = wrapped
    return original


def evaluate(ckpt_path, mode, episodes, test_seed, n_nodes, n_candidates, iql):
    results_config = {
        "accuracy_probability": {"n_episodes": episodes, "episode_length": 30},
        "detection_time": {"n_nodes": n_nodes, "threshold": 0.3},
        "fp_fn": {"n_episodes": episodes, "n_nodes": n_nodes, "threshold": 0.3},
    }
    rng = np.random.default_rng(seed=test_seed + 1)
    original = patch_wire(iql, mode, n_candidates, rng)
    try:
        # 30 rounds, not the 45 of the main table: this is the horizon the
        # paper's listening table was measured at, so compare it column-wise.
        res = iql.test({"number_of_episodes": episodes, "episode_length": 30},
                       results_config, seed=test_seed, max_dead_nodes=2)
    finally:
        iql.network.get_actions_batch = original

    det = res.detection_time
    fp_fn = res.fp_fn.data  # [n_episodes, 2], same accessor run_experiment.py uses
    return {
        "mode": mode,
        "accuracy_probability": float(res.accuracy_probability.probability()),
        "detection_rate": (float(det.n_detected) / det.n_failures) if det.n_failures else float("nan"),
        "mean_detection_time_steps": float(det.mean_detection_time()),
        "n_detected": int(det.n_detected),
        "n_failures": int(det.n_failures),
        "false_positive_rate": float(np.nanmean(fp_fn[:, 0])),
        "false_negative_rate": float(np.nanmean(fp_fn[:, 1])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    iql, network, n, ckpt, n_candidates = build(args.checkpoint, seed=args.test_seed)
    if int(ckpt["n_q_slots"]) == 0:
        raise SystemExit("checkpoint has no Q-slots: nothing on the wire to audit")

    rows = []
    for mode in ("learned", "random", "constant"):
        row = evaluate(args.checkpoint, mode, args.episodes, args.test_seed,
                       n, n_candidates, iql)
        rows.append(row)
        print(f"[{mode:8}] det {row['detection_rate']:.4f} | acc {row['accuracy_probability']:.4f} "
              f"| dt {row['mean_detection_time_steps']:.2f} | FP {row['false_positive_rate']:.4f} "
              f"| FN {row['false_negative_rate']:.4f}", flush=True)

    base = rows[0]["detection_rate"]
    print(f"\ndetection lost by destroying the wire content:")
    for row in rows[1:]:
        print(f"  {row['mode']:8} {base - row['detection_rate']:+.4f} "
              f"({100.0 * (base - row['detection_rate']) / base:+.1f} %)")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "listening_audit.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "episodes": args.episodes,
                   "rows": rows}, f, indent=2)
    print(f"\n[info] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
