import torch
import torch.nn as nn
import timm

class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, layers=3):
        super().__init__()
        mlp = []
        dim = in_dim
        for _ in range(layers - 1):
            mlp.append(nn.Linear(dim, hidden_dim))
            mlp.append(nn.GELU())
            dim = hidden_dim
        mlp.append(nn.Linear(dim, out_dim))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        x = self.mlp(x)
        return nn.functional.normalize(x, dim=-1)


def create_backbones(image_size=128):
    student = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=True,
        num_classes=0,
        img_size=image_size
    )

    teacher = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=True,
        num_classes=0,
        img_size=image_size
    )

    teacher.load_state_dict(student.state_dict())
    return student, teacher


@torch.no_grad()
def ema_update(teacher, student, m):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data = t.data * m + s.data * (1 - m)
