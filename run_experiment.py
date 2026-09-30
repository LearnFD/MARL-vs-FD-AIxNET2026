"""Train and evaluate one detector (see README.md for every reported run).

    python -u run_experiment.py --grid 3 --relay     # hand-coded age relay, L = N
    python -u run_experiment.py --grid 3 \
        --msg-objective counterfactual_td --msg-lr-scale 1.0   # difference reward
    python -u run_experiment.py --grid 3             # own reward, L = diam(G)

The relay and the learned channel differ only in what is on the wire, and
metrics.json records every setting needed to check it. Always run with `python -u`: the
progress line is flushed explicitly, but the rest of stdout is block-buffered
when redirected to a file, which makes a live run look hung.

It trains, evaluates on a held-out test seed, dumps every array behind the
figures to curves.npz, and writes a metrics.json so runs can be tabulated and
compared programmatically.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from network import Network
from env import Environment
from iql import IqlAlgorithms
from grid_adjacency import build_grid_adjacency, hop_distances

# Rounds per training episode. Test episodes are stretched past this by
# --test-episode-scale: detection rate is censored by the horizon (a failure in
# the last rounds cannot be detected in time), and that censoring gets worse as
# the graph diameter grows, so a longer evaluation horizon separates "slow to
# detect" from "never detects".
TRAIN_EPISODE_LENGTH = 30

# Reported regime: 50,000 training episodes, 1,000 held-out episodes stretched
# to 45 rounds. These are defaults rather than required arguments so that a
# clone reproduces the paper without being told the numbers.
REPORTED_EPISODES = 50000
REPORTED_TEST_EPISODES = 1000
REPORTED_TEST_EPISODE_SCALE = 1.5


def build_configs(grid, episodes, relay, test_episodes, q_slots=None,
                  gamma=0.75, neighborhood_reward=0.0, msg_lr_scale=0.1,
                  p_loss_max=0.05, failure_rate=0.01,
                  warmup_steps=500, n_candidates=10, msg_objective="td",
                  test_episode_scale=REPORTED_TEST_EPISODE_SCALE):
    adjacency = build_grid_adjacency(grid, grid)
    n = len(adjacency)
    distances = hop_distances(adjacency)
    message_candidates = list(range(n_candidates))
    # n_age is 0 or N (every age slot is tied to a specific target node); n_q is
    # free-form.
    #
    # The learned width defaults to the graph diameter, so a clone reproduces the
    # reported sizing without having to know the rule: 4 slots at 3x3, 6 at 4x4,
    # 8 at 5x5. The relay's width is set by how many nodes must be *named*, which
    # is what makes it quadratic. A learned code names nothing, so what
    # constrains it instead is how far a symbol's content has to travel before it
    # is acted on -- and the by-distance stratification says reach, not bits per
    # symbol, is what limits this channel. Sizing by diameter keeps the per-agent
    # input at Theta(N * diam), i.e. Theta(N^1.5) on a k x k grid, against the
    # relay's Theta(N^2).
    finite = distances[np.isfinite(distances)]
    diameter = int(finite.max()) if finite.size else 0
    # The relay is the hand-coded arm: age slots only, no learned channel. The
    # learned arm is the reverse.
    n_age = n if relay else 0
    if relay:
        n_q = 0
    elif q_slots is not None:
        n_q = q_slots
    else:
        n_q = diameter
    length_message = n_age + n_q

    env_config = {
        "adjacency_matrix": adjacency,
        "failure_rate": failure_rate,
        "length_message": length_message,
        "message_candidates": message_candidates,
        "neighborhood_reward_weight": neighborhood_reward,
        # Cap on the congestion-driven packet-loss rate (see env.py).
        "p_loss_max": p_loss_max,
    }
    nw_config = {
        "number_of_nodes": n,
        "length_message": length_message,
        "message_candidates": message_candidates,
        "age_ablation": n_age == 0,
    }
    train_config = {
        "learning_rate": 5e-5,
        "exploration": 0.75,
        "update_frequency": 100,
        "target_update_frequency": 2000,
        "number_of_episodes": episodes,
        "episode_length": TRAIN_EPISODE_LENGTH,
        "batch_size": 128,
        "max_buffer_size": 40000,
        "dead_class_weight": 4.0,
        "curriculum_max_initial_failure": 0.25,
        "gamma": gamma,
        "msg_lr_scale": msg_lr_scale,
        "warmup_steps": warmup_steps,
        "msg_objective": msg_objective,
    }
    test_episode_length = int(round(TRAIN_EPISODE_LENGTH * test_episode_scale))
    test_config = {"number_of_episodes": test_episodes, "episode_length": test_episode_length}
    # Belief heatmaps are not stored here: plot_paper_figures.py replays them
    # from the saved weights, which is how the paper's three episodes were drawn.
    test_results_config = {
        "accuracy_probability": {
            "n_episodes": test_episodes,
            "episode_length": test_config["episode_length"],
        },
        "detection_time": {"n_nodes": n, "threshold": 0.3},
        "detection_time_by_distance": {"n_nodes": n, "distances": distances, "threshold": 0.3},
        "beliefs_by_distance": {"n_nodes": n, "distances": distances, "threshold": 0.3},
        "fp_fn": {"n_episodes": test_episodes, "n_nodes": n, "threshold": 0.3},
    }
    return (n, diameter, env_config, nw_config, train_config,
            test_config, test_results_config)


def save_torch_atomic(payload, path):
    """Write a checkpoint so an interrupted write cannot destroy the previous one.

    A periodic checkpoint that a crash can corrupt mid-write is worse than none:
    the run dies *and* takes the last good copy with it. Write to a sibling temp
    file, then rename, which is atomic within a filesystem.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def dump_arrays(path, train_results, test_results):
    """Save every array behind the plots, so figures can be re-drawn from data.

    The paper figures used to be bitmap crops taken out of the rendered contact
    sheets, because nothing kept the arrays. That made vector output, font sizes
    and panel layout impossible to change after a run had finished -- the whole
    point of this dump is that plot_paper_figures.py never has to touch a PNG.
    """
    arrays = {}
    for prefix, obj in (("train", train_results), ("test", test_results)):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            item = getattr(obj, name, None)
            value = getattr(item, "data", None)
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}.{name}.data"] = value

    np.savez_compressed(path, **arrays)
    print(f"[info] arrays -> {path} ({len(arrays)} entries)", flush=True)


def summarize(results, n_q_slots, grid, n, episodes, seed, relay, elapsed):
    """Pull scalar metrics out of the test result object for the comparison table."""
    fp_fn = results.fp_fn.data  # [n_ep, 2]
    det = results.detection_time
    return {
        "grid": f"{grid}x{grid}",
        "n_nodes": int(n),
        "condition": "relay (no Q-slots)" if relay else f"learned ({n_q_slots} Q-slots)",
        "ablation": bool(relay),
        "n_q_slots": int(n_q_slots),
        "train_episodes": int(episodes),
        "train_seed": int(seed),
        "train_time_s": round(float(elapsed), 1),
        "accuracy_probability": float(results.accuracy_probability.probability()),
        "mean_detection_time_steps": float(det.mean_detection_time()),
        "n_detected": int(det.n_detected),
        "n_failures": int(det.n_failures),
        "detection_rate": (float(det.n_detected) / det.n_failures) if det.n_failures else float("nan"),
        "false_positive_rate": float(np.nanmean(fp_fn[:, 0])),
        "false_negative_rate": float(np.nanmean(fp_fn[:, 1])),
        "by_distance": {
            "detection": results.detection_time_by_distance.summary(),
            "beliefs": results.beliefs_by_distance.summary(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate one of the two reported failure detectors. "
                    "Defaults reproduce the learned channel of the paper; add "
                    "--relay for the hand-coded baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- what to run -----------------------------------------------------
    parser.add_argument("--grid", type=int, required=True,
                        help="grid side: 3 gives the 3x3 of the main table, 4 the scaling step")
    parser.add_argument("--relay", action="store_true",
                        help="run the hand-coded age relay instead of the learned channel: "
                             "one heartbeat-age counter per target (L = N), no learned slots")
    parser.add_argument("--folder", type=str, default=None,
                        help="output folder (default: results/<grid>x<grid>_<arm>)")

    # --- size of the run -------------------------------------------------
    parser.add_argument("--episodes", type=int, default=REPORTED_EPISODES,
                        help="training episodes")
    parser.add_argument("--test-episodes", type=int, default=REPORTED_TEST_EPISODES,
                        help="held-out evaluation episodes")
    parser.add_argument("--test-episode-scale", type=float, default=REPORTED_TEST_EPISODE_SCALE,
                        help=f"stretch held-out episodes past the {TRAIN_EPISODE_LENGTH}-round "
                             "training horizon by this factor (1.5 -> 45 rounds). Training is "
                             "unaffected. Detection rate is censored by the horizon, so a longer "
                             "evaluation separates a slow detector from one that never commits. "
                             "Changes the number of failures per episode, so runs at different "
                             "scales are not comparable to each other.")
    parser.add_argument("--seed", type=int, default=42, help="training seed")
    parser.add_argument("--test-seed", type=int, default=123, help="held-out evaluation seed")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <folder>/checkpoint_partial.pt if it exists "
                             "(same command line otherwise). The replay buffer refills and "
                             "the random streams are reseeded from (seed, episode).")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="write checkpoint_partial.pt every N training episodes, so a "
                             "node failure or pre-emption does not throw the run away. "
                             "Holds weights, optimizer states, the episode index and the "
                             "training curves so far. 0 disables it.")

    # --- the message layout ----------------------------------------------
    parser.add_argument("--q-slots", type=int, default=None,
                        help="learned slots. Default is the graph diameter (4 at 3x3, 6 at "
                             "4x4, 8 at 5x5), which keeps per-agent input at Theta(N*diam) "
                             "instead of the relay's Theta(N^2). Ignored with --relay")
    parser.add_argument("--n-candidates", type=int, default=10,
                        help="symbols per slot, i.e. the alphabet size")

    # --- the problem -----------------------------------------------------
    parser.add_argument("--failure-rate", type=float, default=0.01,
                        help="per-round crash probability per alive node")
    parser.add_argument("--p-loss-max", type=float, default=0.05,
                        help="cap on congestion-driven packet loss")

    # --- learning --------------------------------------------------------
    parser.add_argument("--gamma", type=float, default=0.75,
                        help="discount for the message Q-learning (~4 steps of lookahead)")
    parser.add_argument("--msg-lr-scale", type=float, default=0.1,
                        help="message optimizer lr as a multiple of the belief lr")
    parser.add_argument("--msg-objective", choices=("td", "counterfactual_td"), default="td",
                        help="how the message head is trained: one-step Q-learning on the "
                             "agent's own belief error (submitted paper), or per-slot "
                             "Q-learning on a counterfactual difference reward computed on "
                             "the listeners' beliefs at t+1 (second version; run it with "
                             "--msg-lr-scale 1.0)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="steps before the message optimizer starts")
    parser.add_argument("--neighborhood-reward", type=float, default=0.0,
                        help="weight on the mean belief error of the nodes that hear this one. "
                             "0 (reported) means each agent is rewarded for its own accuracy only")

    args = parser.parse_args()

    if args.relay and args.q_slots is not None:
        parser.error("--q-slots sizes the learned channel; --relay has none")

    arm = "relay" if args.relay else "learned"
    if args.folder is None:
        args.folder = os.path.join("results", f"{args.grid}x{args.grid}_{arm}")
    os.makedirs(args.folder, exist_ok=True)

    (n, diameter, env_config, nw_config, train_config,
     test_config, test_results_config) = build_configs(
        args.grid, args.episodes, args.relay, args.test_episodes,
        q_slots=args.q_slots, gamma=args.gamma,
        neighborhood_reward=args.neighborhood_reward,
        msg_lr_scale=args.msg_lr_scale, p_loss_max=args.p_loss_max,
        failure_rate=args.failure_rate,
        warmup_steps=args.warmup_steps,
        n_candidates=args.n_candidates,
        msg_objective=args.msg_objective,
        test_episode_scale=args.test_episode_scale,
    )

    tag = ("RELAY (age slots only)" if args.relay
           else f"LEARNED ({args.q_slots if args.q_slots is not None else diameter} Q-slots)")
    print(f"=== {args.grid}x{args.grid} | {tag} "
          f"| {args.episodes} episodes | seed {args.seed} "
          f"| train rounds {train_config['episode_length']} "
          f"| test rounds {test_config['episode_length']} "
          f"| length_message {nw_config['length_message']} -> {args.folder} ===", flush=True)

    partial_path = os.path.join(args.folder, "checkpoint_partial.pt")
    resume = None
    if args.resume and os.path.exists(partial_path):
        resume = torch.load(partial_path, map_location="cpu", weights_only=False)
        print(f"[resume] {partial_path}: episode {resume['episode']} "
              f"of {resume['total_episodes']}", flush=True)
    elif args.resume:
        print(f"[resume] no {partial_path}; starting from scratch", flush=True)

    # A resumed run reseeds from (seed, episode) so it does not replay the
    # random stream of the first episodes.
    rng = np.random.default_rng(
        seed=args.seed if resume is None else [args.seed, int(resume["episode"])]
    )
    env = Environment(**env_config, rng=rng)
    network = Network(**nw_config, seed=args.seed)
    if resume is not None:
        network.load_checkpoint_state(resume["state_dict"])
    if args.relay:
        width_note = "relay: no learned channel"
    elif args.q_slots is not None:
        width_note = "explicit --q-slots"
    else:
        width_note = "default = graph diameter"
    print(f"[info] n_q_slots = {network.n_q_slots} ({width_note}) "
          f"| n_age_slots = {network.n_age_slots} "
          f"| diameter = {diameter} | device = {network.device}", flush=True)

    iql = IqlAlgorithms(env, network, rng=rng)

    # Metadata shared by the periodic checkpoint and the final one, so a run
    # killed at episode 49,000 leaves a probe-able artifact instead of nothing.
    ckpt_meta = {
        "nw_config": {k: v for k, v in nw_config.items() if k != "adjacency_matrix"},
        "n_q_slots": int(network.n_q_slots),
        "n_belief_slots": int(network.n_belief_slots),
        "n_age_slots": int(network.n_age_slots),
        "grid": args.grid,
        "seed": args.seed,
        "ablation": bool(args.relay),
        "age_ablation": bool(nw_config["age_ablation"]),
    }
    t0 = time.time()

    def write_partial(episode, optimizer_states, diagnostics, train_results):
        payload = dict(ckpt_meta)
        payload.update({
            "state_dict": network.state_dict(),
            "optimizer_states": optimizer_states,
            "episode": int(episode),
            "total_episodes": int(args.episodes),
            "elapsed_s": round(time.time() - t0, 1),
            "diagnostics": diagnostics,
            # The per-episode curves so far; NaN past `episode`. Cheap (a few
            # float32 arrays) and it means a crash costs no analysis either.
            "train_curves": {
                name: getattr(getattr(train_results, name, None), "data", None)
                for name in ("belief_loss", "msg_loss", "mean_chosen_q", "mean_target_q")
            },
        })
        save_torch_atomic(payload, partial_path)
        print(f"[ckpt] episode {episode}/{args.episodes} -> {partial_path}", flush=True)

    train_config["checkpoint_every"] = max(1, args.checkpoint_every)
    train_config["checkpoint_fn"] = None if args.checkpoint_every <= 0 else write_partial
    train_config["resume"] = resume

    train_results = iql.train(train_config)
    elapsed = time.time() - t0

    t_test = time.time()
    test_results = iql.test(
        test_config, test_results_config, seed=args.test_seed, max_dead_nodes=2,
    )
    test_elapsed = time.time() - t_test

    # Every array behind every figure. Plots are drawn afterwards, from this file,
    # by plot_paper_figures.py -- nothing here renders a bitmap.
    dump_arrays(os.path.join(args.folder, "curves.npz"), train_results, test_results)

    # Weights were previously never written anywhere, so every run was
    # unreproducible and nothing could be probed after the fact.
    ckpt_path = os.path.join(args.folder, f"model_seed_{args.seed}.pt")
    save_torch_atomic(
        {
            "state_dict": network.state_dict(),
            **ckpt_meta,
        },
        ckpt_path,
    )
    print(f"[info] weights -> {ckpt_path}", flush=True)

    # The final checkpoint supersedes the crash-insurance copy.
    if os.path.exists(partial_path):
        os.remove(partial_path)

    diagnostics = getattr(iql, "channel_diagnostics", None)
    if diagnostics and diagnostics.get("episode"):
        with open(os.path.join(args.folder, "channel_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)
        tail = min(10, len(diagnostics["episode"]))
        final_spread = float(np.nanmean(diagnostics["q_spread"][-tail:]))
        final_entropy = float(np.nanmean(diagnostics["symbol_entropy_bits"][-tail:]))
        metrics_extra = {
            "final_q_spread": final_spread,
            "final_symbol_entropy_bits": final_entropy,
            "max_symbol_entropy_bits": float(np.log2(len(nw_config["message_candidates"]))),
        }
        print(f"[diag] final q_spread = {final_spread:.4f} "
              f"| final symbol entropy = {final_entropy:.3f} bits "
              f"(max {metrics_extra['max_symbol_entropy_bits']:.2f})", flush=True)
    else:
        metrics_extra = {}

    metrics = summarize(test_results, network.n_q_slots, args.grid, n,
                        args.episodes, args.seed, args.relay, elapsed)
    metrics.update(metrics_extra)
    metrics["gamma"] = args.gamma
    metrics["neighborhood_reward_weight"] = args.neighborhood_reward
    metrics["msg_lr_scale"] = args.msg_lr_scale
    metrics["p_loss_max"] = args.p_loss_max
    metrics["failure_rate"] = args.failure_rate
    # Always 0/True: the hand-coded belief broadcast is gone. Kept in the file so
    # runs from before the cleanup stay field-for-field comparable with runs after.
    metrics["n_belief_slots"] = int(network.n_belief_slots)
    metrics["belief_ablation"] = True
    metrics["age_ablation"] = bool(network.age_ablation)
    metrics["n_age_slots"] = int(network.n_age_slots)
    metrics["detach_message_path"] = bool(network.detach_message_path)
    metrics["warmup_steps"] = int(train_config["warmup_steps"])
    metrics["graph_diameter"] = diameter
    metrics["n_candidates"] = int(args.n_candidates)
    metrics["msg_objective"] = args.msg_objective
    # Episode a resumed run restarted from (None if it ran uninterrupted);
    # train_time_s then counts only the time after the restart.
    metrics["resumed_from_episode"] = (None if resume is None else int(resume["episode"]) - 1)

    # Wall-clock and the hardware it was measured on. A runtime is meaningless
    # without the device, and these runs move between GPU types on the cluster.
    on_cuda = torch.cuda.is_available()
    metrics["train_episode_length"] = int(train_config["episode_length"])
    metrics["test_episode_length"] = int(test_config["episode_length"])
    metrics["test_episode_scale"] = float(args.test_episode_scale)
    metrics["test_episodes"] = int(args.test_episodes)
    metrics["test_time_s"] = round(float(test_elapsed), 1)
    metrics["train_time_per_episode_s"] = round(float(elapsed) / max(1, args.episodes), 5)
    metrics["test_time_per_episode_s"] = round(float(test_elapsed) / max(1, args.test_episodes), 5)
    metrics["device"] = str(network.device)
    metrics["gpu_name"] = torch.cuda.get_device_name(0) if on_cuda else None
    metrics["gpu_count"] = int(torch.cuda.device_count()) if on_cuda else 0
    metrics["torch_version"] = str(torch.__version__)

    with open(os.path.join(args.folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== METRICS ===", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    print("=== BY HOP DISTANCE (observer -> target) ===", flush=True)
    det_by_d = metrics["by_distance"]["detection"]
    bel_by_d = metrics["by_distance"]["beliefs"]
    print(f"{'d':>3s} {'pairs':>8s} {'det_rate':>9s} {'dt_mean':>8s} {'dt_std':>7s} "
          f"{'dt_med':>7s} {'dt_p90':>7s} {'blf_alive':>10s} {'blf_dead':>9s} "
          f"{'gap':>6s} {'acc':>6s}", flush=True)
    for d in sorted(det_by_d, key=int):
        det, bel = det_by_d[d], bel_by_d[d]
        print(f"{d:>3s} {det['n_observer_failure_pairs']:>8d} "
              f"{det['detection_rate']:>9.4f} {det['mean_detection_time_steps']:>8.2f} "
              f"{det['std_detection_time_steps']:>7.2f} {det['median_detection_time_steps']:>7.1f} "
              f"{det['p90_detection_time_steps']:>7.1f} {bel['mean_belief_alive_target']:>10.4f} "
              f"{bel['mean_belief_dead_target']:>9.4f} {bel['belief_gap']:>6.3f} "
              f"{bel['accuracy']:>6.4f}", flush=True)

    print(f"=== Done ({elapsed:.1f}s train + {test_elapsed:.1f}s test "
          f"on {metrics['gpu_name'] or metrics['device']}) ===", flush=True)


if __name__ == "__main__":
    main()
