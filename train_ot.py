import glob
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict

from module import WEMReg, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from ot_merge import lejepa_ot_forward


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    # Anti-collapse regularizer: WEMReg (sliced Gaussian OT) for the ATLAS arms,
    # or SIGReg (LeWM's characteristic-function test) for the LeWM baseline arms.
    reg_name = cfg.loss.sigreg.name
    reg_kwargs = OmegaConf.to_container(cfg.loss.sigreg.kwargs, resolve=True)
    if reg_name == "wemreg":
        regularizer = WEMReg(**reg_kwargs)
    elif reg_name == "sigreg":
        regularizer = SIGReg(**reg_kwargs)
    else:
        raise ValueError(
            f"Unknown regularizer: {reg_name} (expected 'wemreg' or 'sigreg')"
        )

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=regularizer,
        forward=partial(lejepa_ot_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # Persist every logged loss term to run_dir/metrics/version_*/metrics.csv
        logger = CSVLogger(save_dir=str(run_dir), name="metrics", flush_logs_every_n_steps=50)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # Resume across PBS walltime chunks: on PBS there is no SLURM_RESTART_COUNT,
    # so spt.Manager does NOT auto-resume — it needs an explicit absolute
    # ckpt_path to the full-state last.ckpt it wrote under SPT_CACHE_DIR, loaded
    # with weights_only=False (restores epoch/global_step/optimizer/LR). On the
    # first run no last.ckpt exists yet -> resume_ckpt=None -> fresh start.
    resume_ckpt = None
    if os.environ.get("TRAIN_OT_AUTO_RESUME") == "1":
        _cache = os.environ.get("SPT_CACHE_DIR")
        if _cache:
            _cands = sorted(
                glob.glob(os.path.join(_cache, "runs", "*", "*", "*", "checkpoints", "last.ckpt")),
                key=os.path.getmtime,
            )
            if _cands:
                resume_ckpt = _cands[-1]
                print(f"[train_ot] resuming from {resume_ckpt}")
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=resume_ckpt,
        weights_only=False,
    )

    manager()
    return


if __name__ == "__main__":
    run()
