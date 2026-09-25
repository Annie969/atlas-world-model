"""Model-independent in-distribution / out-of-distribution (ID/OOD) split.

The label depends only on the dataset's ground-truth states and the evaluated episode
rows (saved by eval.py), never on any model, so every model's per-episode successes are
split with the SAME labels and are therefore directly comparable.

For each evaluated episode we take the mean, over its expert path
(rows[i] .. rows[i] + offset), of the k-nearest-neighbor distance (k=50) of the true
state to a fixed bank of 50k random training states (StandardScaler + Euclidean).
Episodes above the per-condition median are labeled OOD, the rest ID (100 each for
num_eval=200). This reproduces the ID/OOD split used for the main results.

Usage:
    # after running eval.py, which writes episodes_seed<seed>_off<offset>.npz next to
    # the checkpoint, pass one .npz per seed:
    python idood/id_ood_split.py --task pusht \\
        checkpoints/pusht_atlas/episodes_seed42_off50.npz \\
        checkpoints/pusht_atlas/episodes_seed43_off50.npz ...

The dataset is located via LOCAL_DATASET_DIR or STABLEWM_HOME (a root that contains the
task's HDF5 file, directly or under datasets/); falls back to ~/.stable_worldmodel.
"""
import argparse
import os

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Per-task ground-truth-state source: HDF5 file name + the state column(s) that
# define the OOD space (dataset column names, not model outputs).
TASK_CFG = {
    "pusht": dict(h5="pusht_expert_train.h5", keys=["state"]),
    "tworoom": dict(h5="tworoom.h5", keys=["pos_agent", "pos_target"]),
    "cube": dict(h5="cube_single_expert.h5", keys=["observation"]),
}

BANK_SIZE = 50_000
K = 50
BANK_SEED = 0


def _find_h5(name):
    roots = [
        os.environ.get("LOCAL_DATASET_DIR"),
        os.environ.get("STABLEWM_HOME"),
        os.path.expanduser("~/.stable_worldmodel"),
    ]
    tried = []
    for root in roots:
        if not root:
            continue
        for sub in ("", "datasets", os.path.join("datasets", "ogbench")):
            cand = os.path.join(root, sub, name)
            tried.append(cand)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(
        f"Could not find {name}. Set LOCAL_DATASET_DIR or STABLEWM_HOME. Tried:\n"
        + "\n".join(tried)
    )


def build_bank(task):
    cfg = TASK_CFG[task]
    with h5py.File(_find_h5(cfg["h5"]), "r") as f:
        states = np.concatenate(
            [np.asarray(f[k][:], float).reshape(f[k].shape[0], -1) for k in cfg["keys"]],
            axis=1,
        )
    n = states.shape[0]
    idx = np.sort(np.random.default_rng(BANK_SEED).choice(n, size=min(BANK_SIZE, n), replace=False))
    scaler = StandardScaler().fit(states[idx])
    nn = NearestNeighbors(n_neighbors=K).fit(scaler.transform(states[idx]))
    return scaler, nn, states


def ood_labels(task, rows, offset, bank=None):
    """Return (is_ood: bool[N], path_mean: float[N]) aligned to the given episode rows."""
    scaler, nn, states = bank if bank is not None else build_bank(task)
    spans, flat = [], []
    for r in rows:
        span = np.arange(int(r), int(r) + int(offset) + 1)
        flat.append(span)
        spans.append(len(span))
    flat = np.concatenate(flat)
    dist = nn.kneighbors(scaler.transform(states[flat]))[0].mean(1)
    path_mean = np.empty(len(rows))
    p = 0
    for i, ln in enumerate(spans):
        path_mean[i] = dist[p : p + ln].mean()
        p += ln
    return path_mean > np.median(path_mean), path_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASK_CFG))
    ap.add_argument("eval_npz", nargs="+", help="episodes_seed<seed>_off<offset>.npz from eval.py")
    args = ap.parse_args()

    bank = build_bank(args.task)
    id_rates, ood_rates = [], []
    for fn in args.eval_npz:
        d = np.load(fn)
        rows, succ, offset = d["rows"], d["successes"].astype(bool), int(d["offset"])
        is_ood, _ = ood_labels(args.task, rows, offset, bank)
        idr = 100 * succ[~is_ood].mean()
        oodr = 100 * succ[is_ood].mean()
        id_rates.append(idr)
        ood_rates.append(oodr)
        print(f"{os.path.basename(fn)}: ID={idr:5.1f}%  OOD={oodr:5.1f}%  "
              f"(n_OOD={int(is_ood.sum())}/{len(rows)})")
    if len(id_rates) > 1:
        print(f"\nmean over {len(id_rates)} seeds:  "
              f"ID={np.mean(id_rates):5.2f} +/- {np.std(id_rates):.2f}   "
              f"OOD={np.mean(ood_rates):5.2f} +/- {np.std(ood_rates):.2f}")


if __name__ == "__main__":
    main()
