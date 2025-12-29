import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, img_size):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.final_dim = int(((img_size - 12) / 4) ** 2 * 50)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.final_dim, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.final_dim)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self, img_size):
        self.img_size = img_size
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.final_dim = int(((self.img_size - 12) / 4) ** 2 * 50)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.final_dim, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.final_dim)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class DINOAttention(nn.Module):
    def __init__(self, unfreeze_blocks=0, model_name='dinov2_vitb14'):
        super(DINOAttention, self).__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        self.embed_dim = self.backbone.embed_dim
        self.M = 500
        self.L = 128

        self.feature_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [1, K, 1, H, W]
        x = x.squeeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Dopasowanie do patch_size=14
        if x.shape[-1] % 14 != 0 or x.shape[-2] % 14 != 0:
            x = F.interpolate(x, size=(98, 98), mode='bilinear', align_corners=False)

        H = self.backbone(x)
        H = self.feature_projection(H)

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        Z = torch.mm(A, H)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_objective(self, X, y):
        Y_prob, Y_hat, A = self.forward(X)
        y = y.float().view_as(Y_prob)
        loss = F.binary_cross_entropy(Y_prob, y)
        return loss, Y_prob

    def calculate_classification_error(self, X, Y):
        Y_prob, Y_hat, A = self.forward(X)
        Y = Y.float().view_as(Y_hat)

        error = 1. - Y_hat.eq(Y).float().mean().item()
        return error, Y_hat