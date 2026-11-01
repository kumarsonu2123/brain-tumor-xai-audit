import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.dropout import _DropoutNd


def enable_mc_dropout(model):
    # eval everything (BN uses running stats), then re-enable only dropout. Never model.train().
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, _DropoutNd):
            m.train()
            n += 1
    return n


def mc_state(model):
    bn = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    do = [m for m in model.modules() if isinstance(m, _DropoutNd)]
    return {"n_bn": len(bn), "bn_in_train": sum(m.training for m in bn),
            "n_dropout": len(do), "dropout_in_train": sum(m.training for m in do)}


def assert_mc_mode(model):
    s = mc_state(model)
    assert s["n_dropout"] > 0, "model has no dropout layer -- MC-dropout is meaningless"
    assert s["bn_in_train"] == 0, f"{s['bn_in_train']} BatchNorm layers in train mode"
    assert s["dropout_in_train"] == s["n_dropout"], "some dropout layers are not active"
    return s


def bn_snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def bn_unchanged(before, model):
    after = bn_snapshot(model)
    return before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


@torch.no_grad()
def mc_predict(model, loader, n_passes, device, seed):
    # returns softmax samples (T, N, C); one data load per batch, T passes over it
    assert_mc_mode(model)
    torch.manual_seed(seed)
    out = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        out.append(torch.stack([model(x).float().softmax(1) for _ in range(n_passes)]).cpu())
    return torch.cat(out, dim=1).numpy()


@torch.no_grad()
def deterministic_predict(model, loader, device):
    model.eval()
    return torch.cat([model(b["image"].to(device)).float().softmax(1).cpu() for b in loader]).numpy()