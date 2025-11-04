import torch
from torch import nn
import timm


class SimCLRViT(nn.Module):
    def __init__(self,
                 vit_model_name='vit_tiny_patch16_224.augreg_in21k_ft_in1k',
                 pretrained=True,
                 projection_dim=32,
                 img_size=64):
        super(SimCLRViT, self).__init__()

        self.img_size = img_size

        self.backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            img_size=img_size
        )

        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("ViT model structure not recognized.")

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projection = self.projection_head(features)
        return projection


class ViTWithClassifier(nn.Module):
    def __init__(self, simclr_model, num_classes=10):
        super(ViTWithClassifier, self).__init__()

        self.backbone = timm.create_model(
            simclr_model.backbone.default_cfg['architecture'],
            pretrained=False,
            img_size=simclr_model.img_size
        )

        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError("Backbone structure unexpected.")

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
