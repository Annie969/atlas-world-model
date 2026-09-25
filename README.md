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
- `L_WEMReg`: exact empirical one-dimensional squared Wasserstein distance to `N(0,1)`,
  averaged over time and random projection directions (config key `loss.sigreg`,
  `name: sliced_ot`, weight `λ = 3.0`, `1024` projections).
- `L_OOD`: pairwise-geometry matching against a detached patch-token anchor
  (`loss.ood.weight = μ`).

Setting `μ = 0` (`loss.ood.weight=0.0`) recovers the **LeWM** baseline; `μ = 0.1`
(`loss.ood.weight=0.1`) is **ATLAS**.

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

Train the LeWM baseline (`μ=0`) and ATLAS (`μ=0.1`) for each task. Use `data=pusht`,
`data=tworoom`, or `data=ogb` (OGBench-Cube).

```bash
# PushT
python train_ot.py data=pusht  output_model_name=pusht_lewm    loss.ood.weight=0.0 trainer.devices=1 wandb.enabled=false
python train_ot.py data=pusht  output_model_name=pusht_atlas   loss.ood.weight=0.1 trainer.devices=1 wandb.enabled=false

# TwoRoom
python train_ot.py data=tworoom output_model_name=tworoom_lewm  loss.ood.weight=0.0 trainer.devices=1 wandb.enabled=false
python train_ot.py data=tworoom output_model_name=tworoom_atlas loss.ood.weight=0.1 trainer.devices=1 wandb.enabled=false

# OGBench-Cube
python train_ot.py data=ogb    output_model_name=cube_lewm     loss.ood.weight=0.0 trainer.devices=1 wandb.enabled=false
python train_ot.py data=ogb    output_model_name=cube_atlas    loss.ood.weight=0.1 trainer.devices=1 wandb.enabled=false
```

Weights are written to `<cache>/checkpoints/<output_model_name>/weights_epoch_10.pt`
alongside `config.json`. Train the full 10 epochs so the cosine schedule completes.

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
- `module.py` — `SlicedGaussianOT` (WEMReg), the exact Gaussian quantile-cell sliced OT.
- `ot_merge.py` — objective assembly, including `ood_recovery_loss` (OOD-recovery).
- `jepa.py`, `utils.py` — encoder/predictor model and shared utilities.
- `config/` — Hydra training and evaluation configurations.
- `idood/id_ood_split.py` — model-independent ID/OOD split.
