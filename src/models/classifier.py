import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes=3, dropout_p=0.3, pretrained=True, architecture="resnet18"):
    assert architecture == "resnet18", architecture
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(model.fc.in_features, num_classes))
    return model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    mc = ckpt["model_cfg"]
    model = build_model(num_classes=mc["num_classes"], dropout_p=mc["dropout_p"],
                        pretrained=False, architecture=mc["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt