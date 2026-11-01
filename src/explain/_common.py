import torch


def as_batch(model, image_tensor):
    # dropout on would make every map a random draw; BN in train mode would mix in batch stats
    assert not model.training, "call model.eval() before explaining"
    x = image_tensor if image_tensor.dim() == 4 else image_tensor.unsqueeze(0)
    assert x.shape[0] == 1, "explain() takes one image"
    device = next(model.parameters()).device
    return x.to(device=device, dtype=torch.float32)


def predicted_class(model, x):
    with torch.no_grad():
        return int(model(x).argmax(1))