import argparse
from datetime import datetime
from xmlrpc.client import MAXINT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models.model import SimpleCNNEncoder
from data_utils.dataset_cameleyon import WholeSlideBagDataset

# --- DINO Components ---
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=2048, hidden_dim=1024, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1) # type: ignore
        self.last_layer.weight_g.requires_grad = False # type: ignore

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    def __init__(self, out_dim=2048, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = torch.zeros(1, out_dim).cuda()

    def forward(self, student_output, teacher_output):
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = 0.9 * self.center + 0.1 * batch_center

# --- Augmentation Wrapper ---
class TwoViewTransform:
    """Returns two augmented views of the same image"""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        # x is likely a PIL image or Tensor. Apply transform twice.
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return torch.stack([v1, v2])

def train_dino(args):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # 1. Setup Data
    # Strong augmentations are crucial for DINO
    dino_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.patch_size, scale=(0.4, 1.0)), # Important!
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ElasticTransform()
        transforms.ToTensor(),
    ])
    
    # We use your existing dataset but wrap the transform
    train_ds = WholeSlideBagDataset(
        data_dir=os.path.join(args.data_dir, 'training'),
        transform=TwoViewTransform(dino_transform)
    )
    
    # Batch size 1 because your dataset returns BAGS. We flatten inside the loop.
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Setup Model
    student = SimpleCNNEncoder(img_size=args.patch_size).cuda()
    teacher = SimpleCNNEncoder(img_size=args.patch_size).cuda()
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False # Teacher is frozen

    student_head = DINOHead(in_dim=500).cuda()
    teacher_head = DINOHead(in_dim=500).cuda()
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_head.parameters(): p.requires_grad = False

    optimizer = optim.AdamW(list(student.parameters()) + list(student_head.parameters()), lr=0.0005, weight_decay=0.04)
    criterion = DINOLoss().cuda()

    print(f"Starting DINO pretraining for {args.epochs} epochs...")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} (DINO)')
        
        for batch_idx, (data, _) in enumerate(pbar):
            # data shape is [1, Bag_Size, 2, C, H, W] because Batch_Size=1
            
            # 1. Remove the Batch dimension (squeeze)
            data = data.squeeze(0) # Now shape: [Bag_Size, 2, C, H, W]
            
            # 2. Separate the views
            # view1 takes index 0, view2 takes index 1
            view1 = data[:, 0, :, :, :].cuda() # [Bag_Size, C, H, W]
            view2 = data[:, 1, :, :, :].cuda() # [Bag_Size, C, H, W]

            # OPTIONAL: Subsampling (if you run out of VRAM)
            if view1.size(0) > 64:
                idx = torch.randperm(view1.size(0))[:64]
                view1 = view1[idx]
                view2 = view2[idx]

            # --- Forward Pass (Same as before) ---
            s_feat = student(view1)
            t_feat = teacher(view2)
            
            s_out = student_head(s_feat)
            t_out = teacher_head(t_feat)
            
            loss = criterion(s_out, t_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # EMA Update Teacher
            with torch.no_grad():
                m = 0.99
                for pq, pk in zip(student.parameters(), teacher.parameters()):
                    pk.data.mul_(m).add_((1-m)*pq.detach().data)
                for pq, pk in zip(student_head.parameters(), teacher_head.parameters()):
                    pk.data.mul_(m).add_((1-m)*pq.detach().data)
                criterion.update_center(t_out)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f"total loss: {total_loss/len(train_loader):.4f}")
        if total_loss<best_loss:
            best_loss = total_loss
            # Save Checkpoint
            torch.save(student.state_dict(), f'../data/models/temps/dino_encoder_{timestamp_str}.pth')
            print(f"Saved dino_encoder.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'D:\CAMELEYON16\preprocessed')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=96)
    args = parser.parse_args()
    train_dino(args)