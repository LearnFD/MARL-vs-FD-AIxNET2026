"""Every result figure of the paper, under the file name the paper uses.

Training curves and by-distance panels come from curves.npz,
channel_diagnostics.json and metrics.json (written by run_experiment.py); the
three belief heatmaps are replayed from the saved weights. Conventions:

  * vector PDF, never PNG;
  * no title inside the figure -- the LaTeX caption carries the description;
  * one file per panel, so panels can be re-arranged in LaTeX with subfigure
    rather than frozen into a single composite;
  * large fonts for ticks, axis labels and legends.

Usage (defaults point at the tracked runs in paper_results/):
    python plot_paper_figures.py --out figures
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Categorical slots 1-4 of the reference palette, in fixed order, used unchanged.
# Two-series panels take slots 1 and 2, the most separated pair.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#4a4d52", "#d9d9d6"

# IEEE single-column figures are ~3.5in wide. Type is sized against that, so the
# panels stay legible at final scale without any \resizebox in the paper.
plt.rcParams.update({
    "figure.figsize": (3.6, 2.75),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,          # embed TrueType, so reviewers' viewers match ours
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


PREVIEW = False
# Appended to every file name, e.g. "_v2" for the second version of the paper.
SUFFIX = ""


def finish(fig, ax, out, xlabel, ylabel, legend=False):
    """Common axis dressing. Deliberately sets no title (caption's job)."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.6, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend()
    fig.savefig(out)
    if PREVIEW:
        # PNG copies are for eyeballing layout only; the paper uses the PDFs.
        fig.savefig(out.replace(".pdf", ".preview.png"), dpi=170)
    plt.close(fig)
    print(f"[fig] {out}")


def smooth(y, window):
    """Running mean over a window of episodes; raw per-episode curves at 50k
    episodes are a solid band of noise that hides the trend."""
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window) / window
    valid = np.isfinite(y)
    filled = np.where(valid, y, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def plot_training(npz, diag, out_dir, window):
    """Fig. (training dynamics): one panel per training signal."""
    for key, name, ylabel, color in [
        ("train.belief_loss.data", "belief_loss", "Belief loss", C1),
        ("train.msg_loss.data", "msg_loss", "Message TD loss (MSE)", C2),
    ]:
        y = npz[key]
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(y) + 1), smooth(y, window), color=color)
        finish(fig, ax, os.path.join(out_dir, f"fig_train_{name}{SUFFIX}.pdf"),
               "Training episode", ylabel)

    # chosen vs target Q share one axis: they are the same quantity, and the
    # point of the panel is that they sit on top of each other.
    chosen = npz["train.mean_chosen_q.data"]
    target = npz["train.mean_target_q.data"]
    fig, ax = plt.subplots()
    x = np.arange(1, len(chosen) + 1)
    ax.plot(x, smooth(chosen, window), color=C1, label="chosen")
    ax.plot(x, smooth(target, window), color=C2, label="target", linestyle="--")
    finish(fig, ax, os.path.join(out_dir, f"fig_train_message_q{SUFFIX}.pdf"),
           "Training episode", "Mean message $Q$", legend=True)

    # Q-spread and symbol entropy, sampled every 1,000 episodes by iql.train.
    ep = np.asarray(diag["episode"])
    for key, name, ylabel, color in [
        ("q_spread", "q_spread", "Candidate $Q$ spread", C3),
        ("symbol_entropy_bits", "entropy", "Symbol entropy (bits)", C4),
    ]:
        fig, ax = plt.subplots()
        ax.plot(ep, np.asarray(diag[key], dtype=np.float64), color=color)
        finish(fig, ax, os.path.join(out_dir, f"fig_train_{name}{SUFFIX}.pdf"),
               "Training episode", ylabel)


def replay_heatmaps(checkpoint, episodes, episode_length, test_seed, env_seed):
    """Mean reported p(alive) per node and round, for chosen held-out episodes.

    Replayed from the saved weights rather than read from curves.npz: the
    environment primes each episode's first frame from its own noise stream, so
    beliefs depend on how many resets that stream has seen. A fresh environment
    seeded like the training one (env_seed) and the held-out failure sequence
    (test_seed) is what drew the paper's three panels, cell for cell.
    """
    from audit_listening import build

    iql, _, n, _, _ = build(checkpoint, seed=env_seed)
    config = {"heatmaps": {"n_nodes": n, "episodes_to_store": list(episodes),
                           "episode_length": episode_length}}
    results = iql.test({"number_of_episodes": max(episodes) + 1,
                        "episode_length": episode_length},
                       config, seed=test_seed, max_dead_nodes=2)
    return dict(zip(episodes, results.heatmaps.data))


def plot_heatmaps(heatmaps, out_dir):
    """Mean reported p(alive) per node against round, one stored episode per file.

    The old contact sheet drew this with imshow, which embeds a bitmap inside the
    PDF and defeats the point of vector output. pcolormesh emits one quad per
    cell, so the grid stays sharp at any zoom. The true failure time is drawn on
    top as a black tick at the left edge of the round the node actually died:
    the horizontal distance between that tick and the colour turning red *is*
    the detection delay, which no curve in the paper shows episode by episode.
    """
    # Validated red -> amber -> green ramp, kept from the original heatmaps.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "belief", ["#c0392b", "#f39c12", "#27ae60"]
    )
    for idx, data in heatmaps.items():
        beliefs = data[:, :, 0]
        states = data[:, :, 1]
        valid = int(np.count_nonzero(~np.isnan(beliefs[:, 0])))
        if valid == 0:
            continue
        beliefs, states = beliefs[:valid], states[:valid]
        n_nodes = beliefs.shape[1]

        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        mesh = ax.pcolormesh(
            np.arange(valid + 1), np.arange(n_nodes + 1), beliefs.T,
            cmap=cmap, vmin=0.0, vmax=1.0, shading="flat", rasterized=False,
        )
        # Ground truth drawn on the panel itself rather than in a second one: a
        # black frame spans exactly the rounds the node is really dead. Its left
        # edge is the instant of death, so the horizontal distance from there to
        # where the row turns red *is* the detection delay, readable off the
        # figure. A side-by-side truth panel carries the same information at
        # twice the width, and is almost entirely uniform.
        for node in range(n_nodes):
            dead = np.flatnonzero(states[:, node] == 0)
            if not dead.size:
                continue
            t0 = int(dead[0])
            ax.add_patch(Rectangle(
                (t0, node), valid - t0, 1.0,
                facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3,
            ))
            # Tick on the death instant, so it stays visible where the frame
            # runs along the panel border.
            ax.plot([t0, t0], [node, node + 1], color=INK, linewidth=3.0,
                    solid_capstyle="butt", zorder=4)
        ax.set_yticks(np.arange(n_nodes) + 0.5)
        ax.set_yticklabels(range(n_nodes))
        ax.set_xlabel("Round")
        ax.set_ylabel("Node")
        ax.set_xlim(0, valid)
        ax.set_ylim(0, n_nodes)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label("Mean reported $p(\\mathrm{alive})$", fontsize=10.5)
        bar.outline.set_edgecolor(MUTED)
        out = os.path.join(out_dir, f"fig_heatmap_ep{idx}{SUFFIX}.pdf")
        fig.savefig(out)
        if PREVIEW:
            fig.savefig(out.replace(".pdf", ".preview.png"), dpi=170)
        plt.close(fig)
        print(f"[fig] {out}")


def plot_by_distance(metrics, baseline_metrics, out_dir):
    """Detection and belief quality against observer-to-target hop distance.

    A direct neighbour notices a failure from its own keep-alive, so everything
    beyond distance 1 is what the channel actually bought.
    """
    def series(m, block, field):
        table = m["by_distance"][block]
        ds = sorted(table, key=int)
        return [int(d) for d in ds], [table[d][field] for d in ds]

    for block, field, ylabel, name in [
        ("detection", "detection_rate", "Detection rate", "detection"),
        ("detection", "mean_detection_time_steps", "Detection delay (rounds)", "delay"),
        ("beliefs", "belief_gap", "Belief gap (alive $-$ dead)", "gap"),
    ]:
        x, y = series(metrics, block, field)
        xb, yb = series(baseline_metrics, block, field)
        fig, ax = plt.subplots()
        ax.plot(x, y, color=C1, marker="o", markersize=6, label="learned channel")
        ax.plot(xb, yb, color=C2, marker="s", markersize=6,
                linestyle="--", label="age relay")
        ax.set_xticks(x)
        finish(fig, ax, os.path.join(out_dir, f"fig_dist_{name}{SUFFIX}.pdf"),
               "Hop distance to failed node", ylabel, legend=True)


def load_json(folder, name):
    with open(os.path.join(folder, name)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="paper_results/3x3_learned",
                    help="the learned-channel run folder")
    ap.add_argument("--baseline", default="paper_results/3x3_relay",
                    help="the relay run folder, overlaid on the by-distance panels")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--smooth", type=int, default=500,
                    help="running-mean window in episodes for the training curves")
    ap.add_argument("--heatmap-episodes", type=int, nargs="+", default=[0, 10, 12],
                    help="held-out episodes to draw as belief heatmaps")
    ap.add_argument("--heatmap-rounds", type=int, default=45,
                    help="rounds per replayed episode (the reported test horizon)")
    ap.add_argument("--test-seed", type=int, default=123,
                    help="held-out failure sequence, as in run_experiment.py")
    ap.add_argument("--env-seed", type=int, default=42,
                    help="seed of the replay environment, as in run_experiment.py")
    ap.add_argument("--suffix", default="",
                    help='appended to every file name; "_v2" gives the second '
                         'version of the paper')
    ap.add_argument("--preview", action="store_true",
                    help="also write .preview.png next to each PDF, for checking "
                         "layout without a PDF viewer. Not for the paper.")
    args = ap.parse_args()

    global PREVIEW, SUFFIX
    PREVIEW = args.preview
    SUFFIX = args.suffix

    os.makedirs(args.out, exist_ok=True)

    with np.load(os.path.join(args.run, "curves.npz")) as npz:
        plot_training(npz, load_json(args.run, "channel_diagnostics.json"),
                      args.out, args.smooth)
    plot_by_distance(load_json(args.run, "metrics.json"),
                     load_json(args.baseline, "metrics.json"), args.out)
    plot_heatmaps(
        replay_heatmaps(os.path.join(args.run, "model_seed_42.pt"),
                        args.heatmap_episodes, args.heatmap_rounds,
                        args.test_seed, args.env_seed),
        args.out,
    )


if __name__ == "__main__":
    main()
