import torch


# Match planning-latent geometry to a detached patch-token anchor.
def _norm_pdist(x):
    """Scale-normalized standardized pairwise-distance matrix of x: (N, D) -> (N, N)."""
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    d = torch.cdist(x, x)
    return d / (d.mean() + 1e-6)


def ood_recovery_loss(emb, anchor):
    """L_ood: make emb's pairwise geometry match the OOD-rich anchor's (stop-grad).
    emb, anchor: (B, T, D)."""
    e = emb.reshape(-1, emb.size(-1))
    a = anchor.reshape(-1, anchor.size(-1)).detach()
    return (_norm_pdist(e) - _norm_pdist(a)).pow(2).mean()


def lejepa_ot_forward(self, batch, stage, cfg):
    """encode -> predict -> L_pred + lambda*L_reg + mu*L_ood.

    L_reg is the anti-collapse regularizer selected in train_ot.py:
    WEMReg (ATLAS arms) or SIGReg (LeWM baseline arms)."""
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    mu = float(cfg.loss.get("ood", {}).get("weight", 0.0))

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D) = z
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # prediction loss + anti-collapse regularizer (WEMReg or SIGReg via self.sigreg)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["reg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["reg_loss"]

    # ours: OOD-preserving relational distillation (no-op when mu == 0)
    if mu > 0.0:
        output["ood_loss"] = ood_recovery_loss(emb, output["patch_emb"])
        output["loss"] = output["loss"] + mu * output["ood_loss"]

    # Persist each scalar loss term through the logger.
    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if torch.is_tensor(v) and v.numel() == 1 and "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
