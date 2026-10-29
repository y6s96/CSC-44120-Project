"""Image-caption consistency model (dual encoder).

backbone="small"    : 4-block CNN trained from scratch (the main model)
backbone="resnet18" : ImageNet-pretrained ResNet-18 image encoder (transfer-learning variant)
The text encoder, the score and the losses are the same for both.
"""

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 128
MAX_TOKENS = 24
TOKEN_RE = re.compile(r"[a-z0-9'-]+")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def encode(text, vocab):
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokenize(text)][:MAX_TOKENS]
    return ids + [vocab["<pad>"]] * (MAX_TOKENS - len(ids))


def preprocess(image, size=IMAGE_SIZE, imagenet=False):
    """PIL image -> float tensor (3, size, size). No flipping: it would swap left/right labels."""
    image = image.convert("RGB").resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD if imagenet else (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class ConsistencyModel(nn.Module):
    def __init__(self, vocab_size, dim=256, backbone="small", pretrained=True):
        super().__init__()
        self.backbone = backbone
        if backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            net.fc = nn.Linear(512, dim)
            self.image_encoder = net
        else:
            self.image_encoder = nn.Sequential(
                conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, dim))
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.text_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=1)

    def encode_text(self, tokens):
        mask = (tokens != 0).float().unsqueeze(2)
        mean = (self.embedding(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(self.text_mlp(mean), dim=1)

    def pair_logit(self, img_vec, txt_vec):
        return self.scale * (img_vec * txt_vec).sum(1) + self.bias

    def forward(self, images, tokens):
        """Raw logit; torch.sigmoid(logit) is the consistency score (>= 0.5 = consistent)."""
        return self.pair_logit(self.encode_image(images), self.encode_text(tokens))
