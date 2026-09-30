"""Does the learned channel carry any information about the sender's state?

The entropy already logged during training (`symbol_entropy_bits`) is the *marginal*
entropy H(m) of the emitted symbols. It cannot distinguish the two situations that
matter here:

    a random channel                 -> H(m) high, symbols independent of the state
    a deterministic state-dependent code -> H(m) also high, because states vary

Every diagnosis so far has read "H(m) = 3.2 / 3.32" as "the channel is random". That
inference is not licensed by the measurement. What separates the two cases is the
mutual information

    I(m ; s) = H(m) - H(m | s)

which is zero exactly when the symbol is independent of what the sender knows, and
equal to H(m) when the symbol is a deterministic function of it.

Sender-side descriptors s, all computed on the same rollout:

    neigh_pattern   tuple of true alive/dead flags of the sender's neighbours
                    (the thing a useful protocol would report)
    n_dead_neigh    how many neighbours are dead (coarser, better sampled)
    belief_bin      the sender's own maximum death-belief, in 5 bins
                    (what the sender *thinks*, as opposed to what is true)
    n_dead_global   total dead nodes in the network

Plug-in entropy estimates are biased low for H(m|s) when a descriptor has many
classes and few samples per class, which inflates I. The honest control is the same
statistic on shuffled pairs: `I_shuf` is the noise floor of this estimator on this
sample size, and only I well above it means anything.
"""
import argparse
import json
import os

import numpy as np
import torch

from env import AllNodesDeadError, Environment
from grid_adjacency import build_grid_adjacency
from iql import IqlAlgorithms
from network import Network


def entropy(counts):
    """Shannon entropy in bits of an unnormalised count vector."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(symbols, groups, n_candidates):
    """H(m | groups) in bits, plug-in estimator."""
    h = 0.0
    total = symbols.size
    for g in np.unique(groups):
        sel = groups == g
        counts = np.bincount(symbols[sel], minlength=n_candidates).astype(np.float64)
        h += (sel.sum() / total) * entropy(counts)
    return h


def mutual_information(symbols, groups, n_candidates):
    """I(m ; s) in bits, plug-in estimator, plus H(m) and H(m|s)."""
    h_m = entropy(np.bincount(symbols, minlength=n_candidates).astype(np.float64))
    h_m_given_s = conditional_entropy(symbols, groups, n_candidates)
    return h_m - h_m_given_s, h_m, h_m_given_s


def conditional_mutual_information(symbols, groups, cond, n_candidates):
    """I(m ; s | cond) = H(m | cond) - H(m | cond, s).

    Needed because the sender identity turns out to carry a lot on its own: a
    per-node constant symbol would show mutual information with any descriptor
    that correlates with position (corner/edge nodes do not have the same degree,
    so they cannot have the same neighbour patterns). Conditioning on identity
    strips that static signature out and leaves only the part of the symbol that
    tracks the state *for a given sender*.
    """
    pair = cond.astype(np.int64) * (groups.max() + 1) + groups
    return conditional_entropy(symbols, cond, n_candidates) - conditional_entropy(symbols, pair, n_candidates)


@torch.no_grad()
def collect(iql, adjacency, episodes, episode_length, seed, n_q_slots):
    """Greedy rollout recording every emitted symbol and the sender's state."""
    n = len(adjacency)
    neighbours = [np.where(np.asarray(adjacency[i]) == 1)[0] for i in range(n)]

    rng = iql.get_rng(seed=seed)
    symbols, senders, belief_raw = [], [], []
    desc = {k: [] for k in ("neigh_pattern", "n_dead_neigh", "n_dead_global")}

    iql.network.eval()
    for _ in range(episodes):
        iql.env.reset(forced_rng=rng)
        iql._reset_protocol_state()
        actions = iql.initialize_actions()

        for step_index in range(episode_length):
            alive_nodes = np.where(iql.env.states == 1)[0]
            if alive_nodes.size == 0:
                break

            frames = iql.build_frames(alive_nodes, init=(step_index == 0))
            belief_probs, q_ids, h_out = iql.network.get_actions_batch(
                frames, iql.hiddens[alive_nodes], alive_nodes
            )
            iql.hiddens[alive_nodes] = h_out

            states = iql.env.states
            n_dead_global = int((states == 0).sum())
            for k, node in enumerate(alive_nodes):
                nb = neighbours[node]
                dead_flags = (states[nb] == 0).astype(np.int64)
                # Pack the neighbour death pattern into one integer label.
                pattern = int((dead_flags * (2 ** np.arange(dead_flags.size))).sum())
                # The sender's own view: strongest death-belief it holds about
                # anyone else. Stored raw and binned by empirical quantiles after
                # collection -- a fixed grid put every sample in one bin.
                own = np.delete(belief_probs[k], node)
                belief_raw.append(float(1.0 - own.min()))

                symbols.append(q_ids[k])
                senders.append(int(node))
                desc["neigh_pattern"].append(pattern)
                desc["n_dead_neigh"].append(int(dead_flags.sum()))
                desc["n_dead_global"].append(n_dead_global)

            iql.write_actions(actions, alive_nodes, belief_probs, q_ids)
            dead_nodes = np.where(iql.env.states == 0)[0]
            actions[dead_nodes, :] = iql.dead_action
            try:
                iql.env.forward(actions, forced_rng=rng)
            except AllNodesDeadError:
                break
            iql._update_ages()

    desc = {k: np.asarray(v, dtype=np.int64) for k, v in desc.items()}

    # Quantile bins, so the descriptor is well populated whatever the belief scale.
    raw = np.asarray(belief_raw, dtype=np.float64)
    edges = np.unique(np.quantile(raw, [0.2, 0.4, 0.6, 0.8]))
    desc["belief_bin"] = np.digitize(raw, edges)

    return (np.asarray(symbols, dtype=np.int64).reshape(-1, n_q_slots),
            np.asarray(senders, dtype=np.int64), desc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="model_seed_*.pt written by run_experiment.py")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=45,
                        help="rounds per episode; 45 is the reported test horizon and "
                             "reproduces the paper's channel_mi.json exactly")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", type=str, default=None, help="write JSON here (default: alongside the checkpoint)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    grid = ckpt["grid"]
    adjacency = build_grid_adjacency(grid, grid)
    n = len(adjacency)

    # nw_config was saved without adjacency_matrix (Network does not take one).
    nw_config = dict(ckpt["nw_config"])
    network = Network(**nw_config, seed=ckpt["seed"])
    dropped = network.load_checkpoint_state(ckpt["state_dict"])
    if dropped:
        print(f"[info] dropped {dropped} legacy trust-gate tensors "
              "(never read by forward; see Network.load_checkpoint_state)")
    network.eval()

    n_candidates = len(nw_config["message_candidates"])
    n_q_slots = int(ckpt["n_q_slots"])
    if n_q_slots == 0:
        raise SystemExit("checkpoint has no Q-slots: nothing to measure")

    # p_loss/failure_rate only shape the rollout distribution; the defaults match
    # the training runs this is meant to probe.
    env = Environment(
        adjacency_matrix=adjacency, failure_rate=0.01,
        length_message=nw_config["length_message"],
        message_candidates=nw_config["message_candidates"],
        p_loss_max=0.05, rng=np.random.default_rng(seed=args.seed),
    )
    iql = IqlAlgorithms(env, network, rng=np.random.default_rng(seed=args.seed))

    print(f"[info] grid {grid}x{grid} | n {n} | q_slots {n_q_slots} x {n_candidates} symbols "
          f"| max H(m) = {np.log2(n_candidates):.3f} bits", flush=True)

    symbols, senders, desc = collect(
        iql, adjacency, args.episodes, args.episode_length, args.seed, n_q_slots
    )
    print(f"[info] {symbols.shape[0]} emissions collected", flush=True)

    shuffler = np.random.default_rng(seed=args.seed + 1)
    report = {
        "checkpoint": args.checkpoint,
        "n_emissions": int(symbols.shape[0]),
        "n_q_slots": n_q_slots,
        "n_candidates": n_candidates,
        "max_entropy_bits": float(np.log2(n_candidates)),
        "slots": [],
    }

    for slot in range(n_q_slots):
        m = symbols[:, slot]
        entry = {"slot": slot, "descriptors": {}}
        for name, s in desc.items():
            mi, h_m, h_m_given_s = mutual_information(m, s, n_candidates)
            mi_shuf, _, _ = mutual_information(m, shuffler.permutation(s), n_candidates)
            cmi = conditional_mutual_information(m, s, senders, n_candidates)
            # Shuffling s *within* each sender is the matching control for the
            # conditional statistic: it destroys the state dependence while
            # leaving the identity signature intact.
            s_shuf_within = s.copy()
            for node in np.unique(senders):
                sel = senders == node
                s_shuf_within[sel] = shuffler.permutation(s[sel])
            cmi_shuf = conditional_mutual_information(m, s_shuf_within, senders, n_candidates)
            entry["H_m_bits"] = h_m
            entry["descriptors"][name] = {
                "I_bits": mi, "H_m_given_s_bits": h_m_given_s,
                "I_shuffled_bits": mi_shuf, "n_classes": int(np.unique(s).size),
                "I_given_sender_bits": cmi, "I_given_sender_shuffled_bits": cmi_shuf,
            }
        # Sender identity is not "information about the state", but a code that
        # differs per node would still show up as MI against every descriptor, so
        # it has to be measured to be ruled out.
        mi_id, _, _ = mutual_information(m, senders, n_candidates)
        mi_id_shuf, _, _ = mutual_information(m, shuffler.permutation(senders), n_candidates)
        entry["descriptors"]["sender_id"] = {
            "I_bits": mi_id, "H_m_given_s_bits": entry["H_m_bits"] - mi_id,
            "I_shuffled_bits": mi_id_shuf, "n_classes": int(np.unique(senders).size),
        }
        report["slots"].append(entry)

    cols = ("neigh_pattern", "n_dead_neigh", "belief_bin", "n_dead_global")
    for title, keys in (("I(m ; s)", ("I_bits", "I_shuffled_bits")),
                        ("I(m ; s | sender)", ("I_given_sender_bits", "I_given_sender_shuffled_bits"))):
        header = f"{'slot':>4} {'H(m)':>7} " + " ".join(f"{k:>16}" for k in cols)
        print(f"\n=== {title} ===")
        print(header)
        print("-" * len(header))
        for entry in report["slots"]:
            cells = [f"{entry['descriptors'][k][keys[0]]:.3f}/{entry['descriptors'][k][keys[1]]:.3f}"
                     for k in cols]
            print(f"{entry['slot']:>4} {entry['H_m_bits']:7.3f} " + " ".join(f"{c:>16}" for c in cells))
    print("\nI(m ; sender_id) per slot: " +
          "  ".join(f"{e['descriptors']['sender_id']['I_bits']:.3f}" for e in report["slots"]))
    print("\n(value / control, in bits. The control is the same statistic on shuffled")
    print(" labels -- the estimator's noise floor at this sample size. Only a value")
    print(" clearly above its control means the symbol depends on that descriptor.")
    print(" The conditional table is the one that matters: it removes the static")
    print(" per-sender signature, which is large here.)")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "channel_mi.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[info] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
