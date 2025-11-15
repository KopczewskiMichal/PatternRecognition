import torch
from torch import nn
from torch.nn import functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        student_output: [B * (2+local), D]
        teacher_output: [B * 2, D]
        """
        batch_size = teacher_output.shape[0] // 2
        views_per_sample = student_output.shape[0] // batch_size

        # Normalizacja
        student_pred = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_pred = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()

        # Reshape
        student_pred = student_pred.view(batch_size, views_per_sample, -1)  # [B, V, D]
        teacher_pred = teacher_pred.view(batch_size, 2, -1)  # [B, 2, D]

        # Loss wektorowo - unikamy pętli
        # [B, 2, D] -> [B, 2, 1, D] i [B, V, D] -> [B, 1, V, D]
        teacher_expanded = teacher_pred.unsqueeze(2)  # [B, 2, 1, D]
        student_expanded = student_pred.unsqueeze(1)  # [B, 1, V, D]

        # Cross entropy: -sum(q * log(p))
        losses = -torch.sum(teacher_expanded * student_expanded, dim=-1)  # [B, 2, V]
        total_loss = losses.mean()

        # Update center
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss
