import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, img_size, M_size = 500, L_size = 128):
        super(Attention, self).__init__()
        self.M = M_size
        self.L = L_size
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
            nn.BatchNorm1d(self.L),
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

    
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Encoder (Shared between DINO and MIL) ---
class SimpleCNNEncoder(nn.Module):
    def __init__(self, img_size=96):
        super().__init__()
        self.img_size = img_size
        # Calculate final dim based on your original logic
        self.final_dim = int(((self.img_size - 12) / 4) ** 2 * 50)
        self.M = 500

        self.part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5), # Changed input to 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.part2 = nn.Sequential(
            nn.Linear(self.final_dim, self.M),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.part1(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.part2(x)         # Output dim: 500
        return x

# --- 2. The Full MIL Model (Wraps the Encoder) ---
class GatedAttention(nn.Module):
    def __init__(self, encoder, attention_branches=1):
        super().__init__()
        self.feature_extractor = encoder 
        self.L = 128
        self.ATTENTION_BRANCHES = attention_branches
        self.M = 500 

        self.attention_V = nn.Sequential(nn.BatchNorm1d(self.M),nn.Linear(self.M, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.BatchNorm1d(self.M),nn.Linear(self.M, self.L), nn.Sigmoid())
        self.attention_w = nn.Sequential(nn.BatchNorm1d(self.L),nn.Linear(self.L, self.ATTENTION_BRANCHES))
        self.classifier = nn.Sequential(nn.Linear(self.M*self.ATTENTION_BRANCHES, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(0)
        # x input shape: [Bag_Size, C, H, W]
        H = self.feature_extractor(x) # [Bag_Size, 500]

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        Z = torch.mm(A, H) 
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        
        return Y_prob, Y_hat, A

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat