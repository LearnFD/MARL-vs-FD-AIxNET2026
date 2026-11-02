"""Does mutual information rank working channels? (It does not.)

Exploratory check, not reported in the final paper: across the
non-degenerate 3x3 conditions, per-slot mutual information correlates weakly
with detection rate -- MI is evidence the messages carry state, not a proxy
for detector quality. The paper makes the same point with the frozen sender
(less MI than the own-reward head, same detection).

Pairs every run directory that has both a channel_mi.json (written by
analyze_channel_mi.py) and a metrics.json (written by run_experiment.py),
then correlates mean I(m;state|sender) against detection rate.

Two exclusions, both properties of the comparison rather than of the scores:

  * A run whose symbol entropy has collapsed is not emitting a code at all, so
    asking whether its MI predicts its detection rate is meaningless.
  * Runs are held to a single training length. Detection rate improves with
    training while MI does not move with it (the reported configuration gains
    detection between 10k and 50k episodes while its MI *falls*), so mixing
    regimes measures training length, not the MI-detection relationship.

Both thresholds are printed with the result, so the exclusions can be checked
rather than trusted. On the shipped runs the defaults give ten non-degenerate
3x3 conditions, Pearson +0.16, Spearman -0.04.

Usage (from the repository root):
    python exploratory/correlate_mi.py exploratory/mi_correlation
    python exploratory/correlate_mi.py exploratory/mi_correlation --train-episodes 50000
"""
import argparse
import glob
import json
import os

import numpy as np

# Default descriptor: the alive/dead pattern of the sender's neighbours, which is
# what the paper measures the symbols against.
DEFAULT_DESCRIPTOR = "neigh_pattern"

# A channel emitting fewer than this many bits per slot has collapsed to a near
# constant symbol; see the entropy diagnostics in iql.py.
ENTROPY_FLOOR_BITS = 0.5


def spearman(x, y):
    """Rank correlation, computed here to avoid a scipy dependency.

    Ties get averaged ranks, which matters because detection rates repeat
    across near-identical arms.
    """
    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # Average the ranks inside each group of equal values.
        for value in np.unique(v):
            same = v == value
            if same.sum() > 1:
                r[same] = r[same].mean()
        return r
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


def collect(root, descriptor):
    rows = []
    for mi_path in sorted(glob.glob(os.path.join(root, "*", "channel_mi.json"))):
        run = os.path.dirname(mi_path)
        metrics_path = os.path.join(run, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        mi = json.load(open(mi_path))
        metrics = json.load(open(metrics_path))

        per_slot = [s["descriptors"][descriptor]["I_given_sender_bits"]
                    for s in mi["slots"] if descriptor in s["descriptors"]]
        if not per_slot:
            continue
        floors = [s["descriptors"][descriptor]["I_given_sender_shuffled_bits"]
                  for s in mi["slots"] if descriptor in s["descriptors"]]

        rows.append({
            "run": os.path.basename(run),
            "grid": metrics.get("grid"),
            "train_episodes": metrics.get("train_episodes"),
            "mi_bits": float(np.mean(per_slot)),
            "mi_floor_bits": float(np.mean(floors)),
            "detection_rate": metrics.get("detection_rate"),
            "accuracy": metrics.get("accuracy_probability"),
            "entropy_bits": metrics.get("final_symbol_entropy_bits"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding the run folders")
    ap.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR,
                    help=f"sender-state descriptor to correlate (default {DEFAULT_DESCRIPTOR})")
    ap.add_argument("--grid", default="3x3",
                    help="restrict to one grid size; 'all' to keep every run")
    ap.add_argument("--train-episodes", type=int, default=10000,
                    help="restrict to one training length, since detection rate depends "
                         "on it and MI does not; 0 to keep every run")
    ap.add_argument("--entropy-floor", type=float, default=ENTROPY_FLOOR_BITS,
                    help="drop runs whose final symbol entropy is below this (collapsed channel)")
    args = ap.parse_args()

    rows = collect(args.root, args.descriptor)
    if args.grid != "all":
        rows = [r for r in rows if r["grid"] == args.grid]
    off_regime = []
    if args.train_episodes:
        off_regime = [r for r in rows if r["train_episodes"] != args.train_episodes]
        rows = [r for r in rows if r["train_episodes"] == args.train_episodes]

    kept, dropped = [], []
    for r in rows:
        ent = r["entropy_bits"]
        (dropped if (ent is None or ent < args.entropy_floor) else kept).append(r)

    print(f"descriptor: {args.descriptor} | grid: {args.grid} | "
          f"train episodes: {args.train_episodes or 'any'} | "
          f"entropy floor: {args.entropy_floor} bits")
    print(f"{len(rows)} runs in regime, {len(kept)} non-degenerate, {len(dropped)} dropped\n")

    print(f"{'run':28s} {'MI b/slot':>10s} {'floor':>8s} {'entropy':>8s} {'det':>7s} {'acc':>7s}")
    for r in sorted(kept, key=lambda r: -r["mi_bits"]):
        print(f"{r['run']:28s} {r['mi_bits']:>10.4f} {r['mi_floor_bits']:>8.4f} "
              f"{r['entropy_bits']:>8.3f} {r['detection_rate']:>7.4f} {r['accuracy']:>7.4f}")
    if dropped:
        print("\ndropped as degenerate (collapsed symbol entropy):")
        for r in sorted(dropped, key=lambda r: r["run"]):
            ent = "n/a" if r["entropy_bits"] is None else f"{r['entropy_bits']:.3f}"
            print(f"  {r['run']:28s} entropy {ent}")
    if off_regime:
        print(f"\noutside the {args.train_episodes}-episode regime, so not comparable:")
        for r in sorted(off_regime, key=lambda r: r["run"]):
            print(f"  {r['run']:28s} {r['train_episodes']} episodes, "
                  f"MI {r['mi_bits']:.4f}, det {r['detection_rate']:.4f}")

    if len(kept) < 3:
        raise SystemExit("\nnot enough non-degenerate runs to correlate")

    mi = [r["mi_bits"] for r in kept]
    det = [r["detection_rate"] for r in kept]
    acc = [r["accuracy"] for r in kept]
    print(f"\nn = {len(kept)}")
    print(f"MI vs detection : Pearson {np.corrcoef(mi, det)[0, 1]:+.2f}  "
          f"Spearman {spearman(mi, det):+.2f}")
    print(f"MI vs accuracy  : Pearson {np.corrcoef(mi, acc)[0, 1]:+.2f}  "
          f"Spearman {spearman(mi, acc):+.2f}")


if __name__ == "__main__":
    main()
