import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finish():
    # ToDtype with a bare dtype only touches images, so masks stay uint8
    return [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def get_train_transforms(image_size=224, hflip_p=0.5, rotation_deg=10, brightness=0.1):
    # resize first: rotating a 512px image only to downsample it afterwards is wasted work
    return v2.Compose([
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=hflip_p),
        v2.RandomRotation(degrees=rotation_deg),
        v2.ColorJitter(brightness=brightness),
        *_finish(),
    ])


def get_eval_transforms(image_size=224):
    return v2.Compose([v2.Resize((image_size, image_size), antialias=True), *_finish()])


def get_stability_transform(spec, image_size=224):
    # spec is one entry of cfg["audit"]["stability_transforms"]
    if spec["type"] == "rotate":
        t = v2.RandomRotation(degrees=(spec["degrees"], spec["degrees"]))
    elif spec["type"] == "brightness":
        t = v2.ColorJitter(brightness=(spec["factor"], spec["factor"]))
    else:
        raise ValueError(spec)
    return v2.Compose([v2.Resize((image_size, image_size), antialias=True), t, *_finish()])