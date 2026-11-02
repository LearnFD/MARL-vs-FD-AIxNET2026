# LearnFD: learned cooperative failure detection

Code and results for *Learning to Suspect: Emergent Communication for
Decentralized Failure Detection under Partial Observability* (AIxNET 2026).

Agents on a grid must find out which peers have crashed. They see only their
neighbours, so news about distant nodes has to travel through messages. Each
message is a few discrete symbols chosen by Q-learning; a hand-coded
heartbeat-age relay is the baseline.

## Setup

```bash
uv sync                               # or: pip install torch numpy matplotlib
```

A GPU is optional but much faster (one 3×3 run takes about an hour on an L40S).

## Reproduce the results

Every number of the paper is in `paper_results/*/metrics.json`, and the
figures are in `figures/`. To retrain a run (results go to `results/`):

| Run | Command | In `paper_results/` |
|---|---|---|
| Hand-coded relay | `python -u run_experiment.py --grid 3 --relay` | `3x3_relay` |
| Learned, difference reward | `python -u run_experiment.py --grid 3 --msg-objective counterfactual_td --msg-lr-scale 1.0` | `3x3_learned_dr` |
| Learned, own reward | `python -u run_experiment.py --grid 3` | `3x3_learned` |
| Frozen sender (no message training) | `python -u run_experiment.py --grid 3 --msg-lr-scale 0` | `3x3_frozen` |
| Own reward, higher learning rate | `python -u run_experiment.py --grid 3 --msg-lr-scale 1.0` | `3x3_learned_lr1` |

Add `--seed 43` or `--seed 44` for the other seeds (`*_s43`, `*_s44`), and use
`--grid 4` for the 4×4 runs (`4x4_*`; `--q-slots 4` for `4x4_learned_L4`).
An interrupted run continues with the same command plus `--resume`.

Analyses, from the saved weights (`--out` avoids overwriting the shipped
files):

```bash
python analyze_channel_mi.py --checkpoint paper_results/3x3_learned_dr/model_seed_42.pt --out mi.json
python audit_listening.py    --checkpoint paper_results/3x3_learned_dr/model_seed_42.pt --out audit.json
```

Figures: `figures/` holds the paper's figures (difference reward) and
`figures/own_reward/` the same figures for the own-reward run. To redraw them:

```bash
python plot_paper_figures.py --out figures
python plot_paper_figures.py --run paper_results/3x3_learned --out figures/own_reward
```

`exploratory/` holds earlier 10,000-episode runs that the paper does not
report; see `exploratory/README.md`.

Training on a GPU is not bit-reproducible (CUDA's embedding backward is
non-deterministic); on the CPU it is.

## License and citation

MIT license. To cite this work, see `CITATION.cff`.
