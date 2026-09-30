# ATLAS: Aligned Transport of Latent Structure for Reliable World-Model Planning

Reference implementation of **ATLAS**, a joint-embedding predictive (JEPA) world model
that makes two changes to the LeWM training objective so the planning latent stays
informative about when a state is out-of-distribution (OOD):

1. **WEMReg** (Wasserstein embedding matching): an exact sliced Gaussian optimal-transport
   term that replaces the characteristic-function normality test (SIGReg) used to prevent
   latent collapse.
2. **OOD-recovery**: a relational-distillation term that matches the planning latent's
   scale-free pairwise-distance geometry to a stop-gradient patch-token anchor.

This repository contains the code to reproduce the main planning-success results for
**LeWM** and **ATLAS**, including the model-independent ID/OOD split.

## Objective

```text
L = L_pred + λ · L_WEMReg + μ · L_OOD
```

- `L_pred`: next-step latent prediction loss.
- `L_reg`: the anti-collapse regularizer, selected by the `reg` config group:
  - `reg=wemreg` — **WEMReg**, an exact empirical one-dimensional squared Wasserstein
    distance to `N(0,1)` averaged over `1024` random projections (weight `λ = 3.0`). Used by ATLAS.
  - `reg=sigreg` — **SIGReg**, the LeWM baseline's characteristic-function (Epps–Pulley) test
    (weight `λ = 0.09`). Used by LeWM.
- `L_OOD`: OOD-recovery, pairwise-geometry matching against a detached patch-token anchor
  (`loss.ood.weight = μ`, `0` = off, `0.1` = on).

The two switches give the `2×2` ablation `{SIGReg, WEMReg} × {OOD off, on}`:

| model | `reg` | `loss.ood.weight` |
|---|---|---|
| **LeWM** (baseline) | `sigreg` | `0.0` |
| LeWM + OOD | `sigreg` | `0.1` |
| WEMReg | `wemreg` | `0.0` |
| **ATLAS** (ours) | `wemreg` | `0.1` |

## Setup

```bash
pip install "stable-worldmodel[train,env]" stable-pretraining
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install hydra-core lightning h5py hdf5plugin einops scikit-learn

# Root holding datasets/ and checkpoints/ (used by training, evaluation, and the split).
export STABLEWM_HOME=/path/to/stablewm
# Optional separate dataset cache root containing the task HDF5 files (or a datasets/ folder).
export LOCAL_DATASET_DIR=/path/to/data
```

Tasks: **PushT**, **TwoRoom**, and **OGBench-Cube**, all trained and evaluated inside the
`stable-worldmodel` harness on its shipped offline demonstration datasets. Shared settings:
history length 3, frameskip 5, 10 training epochs, batch size 128, AdamW at `5e-5` with a
cosine schedule, training seed 3072.

## Train

Train **LeWM** (`reg=sigreg`) and **ATLAS** (`reg=wemreg    loss.ood.weight=0.1`) for each
task. Use `data=pusht`, `data=tworoom`, or `data=ogb` (OGBench-Cube).

```bash
# PushT
python train_ot.py data=pusht  reg=sigreg    loss.ood.weight=0.0 output_model_name=pusht_lewm  trainer.devices=1 wandb.enabled=false
python train_ot.py data=pusht  reg=wemreg    loss.ood.weight=0.1 output_model_name=pusht_atlas trainer.devices=1 wandb.enabled=false

# TwoRoom
python train_ot.py data=tworoom reg=sigreg    loss.ood.weight=0.0 output_model_name=tworoom_lewm  trainer.devices=1 wandb.enabled=false
python train_ot.py data=tworoom reg=wemreg    loss.ood.weight=0.1 output_model_name=tworoom_atlas trainer.devices=1 wandb.enabled=false

# OGBench-Cube
python train_ot.py data=ogb    reg=sigreg    loss.ood.weight=0.0 output_model_name=cube_lewm  trainer.devices=1 wandb.enabled=false
python train_ot.py data=ogb    reg=wemreg    loss.ood.weight=0.1 output_model_name=cube_atlas trainer.devices=1 wandb.enabled=false
```

For the other two ablation arms, use `reg=sigreg loss.ood.weight=0.1` (LeWM + OOD) and
`reg=wemreg loss.ood.weight=0.0` (WEMReg). Weights are written to
`<cache>/checkpoints/<output_model_name>/weights_epoch_10.pt` alongside `config.json`.
Train the full 10 epochs so the cosine schedule completes.

## Evaluate (goal offset 50, main results)

Evaluate each checkpoint at goal offset 50 for seeds 42–46. Use `--config-name=pusht`,
`tworoom`, or `cube`. The evaluation budget is `offset + 25 = 75`; planning uses CEM with
300 candidates, 30 iterations, and 30 elites (`config/eval/solver/cem.yaml`).

```bash
for seed in 42 43 44 45 46; do
  python eval.py --config-name=pusht \
    policy=pusht_atlas/weights_epoch_10.pt \
    eval.num_eval=200 eval.goal_offset_steps=50 eval.eval_budget=75 \
    solver=cem solver.device=cuda seed=$seed
done
```

Each run reports the overall `success_rate` and writes
`episodes_seed<seed>_off50.npz` (per-episode successes aligned to the evaluated episode
rows) next to the checkpoint. Repeat for `pusht_lewm` and for the TwoRoom / Cube
checkpoints with their config names.

## ID/OOD split (reproduce the main figure)

Split each condition's 200 episodes into in-distribution and out-of-distribution halves
with a fixed, model-independent rule (median expert-path k-NN distance to a 50k
training-state bank), then read off per-half success. Pass the per-seed `.npz` files that
`eval.py` wrote:

```bash
python idood/id_ood_split.py --task pusht \
  <cache>/checkpoints/pusht_atlas/episodes_seed42_off50.npz \
  <cache>/checkpoints/pusht_atlas/episodes_seed43_off50.npz \
  <cache>/checkpoints/pusht_atlas/episodes_seed44_off50.npz \
  <cache>/checkpoints/pusht_atlas/episodes_seed45_off50.npz \
  <cache>/checkpoints/pusht_atlas/episodes_seed46_off50.npz
```

This prints ID and OOD success per seed and the mean ± std over seeds. Running it for the
LeWM and ATLAS checkpoints of each task reproduces the LeWM and ATLAS entries of the main
ID/OOD results.

## Files

- `train_ot.py` — training entry point (`L_pred + λ·L_WEMReg + μ·L_OOD`).
- `eval.py` — zero-shot CEM planning evaluation; saves per-episode successes.
- `module.py` — `WEMReg` (sliced Gaussian OT) and `SIGReg` (LeWM baseline regularizer).
- `ot_merge.py` — objective assembly, including `ood_recovery_loss` (OOD-recovery).
- `config/train/reg/` — the `wemreg` (WEMReg) and `sigreg` (LeWM) regularizer configs.
- `jepa.py`, `utils.py` — encoder/predictor model and shared utilities.
- `config/` — Hydra training and evaluation configurations.
- `idood/id_ood_split.py` — model-independent ID/OOD split.
