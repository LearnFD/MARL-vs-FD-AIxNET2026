# Exploratory runs

Runs from the development of the method, kept for transparency. **None of these
numbers is reported in the paper**; the paper's results are in
`paper_results/`.

Each folder in `mi_correlation/` holds the `metrics.json` and `channel_mi.json`
of one 3×3 configuration trained for 10,000 episodes (`qslots_4x4` is 4×4;
`qslots_50k_3x3` is the 50,000-episode own-reward run, the same run as
`paper_results/3x3_learned`). Their settings, recorded in each `metrics.json`,
differ from the final ones (for example γ = 0.95, a neighborhood reward and a
3,000-step warmup), and their `condition` labels come from an earlier version of
the code. The `checkpoint` paths inside `channel_mi.json` give the runs'
original location; the weights are not shipped.

`correlate_mi.py` uses them to check whether mutual information ranks working
channels. It does not (Pearson +0.16, Spearman −0.04 against detection rate):

```bash
python exploratory/correlate_mi.py exploratory/mi_correlation
```
