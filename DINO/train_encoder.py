import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from tqdm import tqdm

from models import create_backbones, DinoHead, ema_update
from dino_loss import DINOLoss

device = "cuda"

image_size = 32
local_size = 16
local_crops = 6
batch_size = 64
epochs = 30
momentum_base = 0.996
lr = 1e-4
weight_decay = 1e-5

def create_transforms(img, image_size=image_size, local_size=local_size, local_views=local_crops):
    global_t = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    local_t = transforms.Compose([
        transforms.RandomResizedCrop(local_size, scale=(0.1, 0.5)),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    global_imgs = [global_t(img), global_t(img)]
    local_imgs = [local_t(img) for _ in range(local_views)]
    return global_imgs + local_imgs

def save_checkpoint(student, teacher, student_head, teacher_head, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'student_head_state_dict': student_head.state_dict(),
        'teacher_head_state_dict': teacher_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    dataset = FashionMNIST(root="./../data", download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    student, teacher = create_backbones(image_size)
    student_head = DinoHead(student.num_features).to(device)
    teacher_head = DinoHead(teacher.num_features).to(device)

    teacher.load_state_dict(student.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())

    student = student.to(device)
    teacher = teacher.to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = DINOLoss().to(device)

    best_loss = 1000000
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(pbar):
            all_views = []

            for img in images:
                img_pil = transforms.ToPILImage()(img)
                views = create_transforms(img_pil)
                all_views.extend(views)

            x = torch.stack(all_views, dim=0).to(device)

            student_out = student_head(student(x))

            total_views_per_sample = 2 + local_crops
            global_indices = []
            for i in range(0, x.shape[0], total_views_per_sample):
                global_indices.extend([i, i + 1])
            teacher_x = x[global_indices]

            with torch.no_grad():
                teacher_out = teacher_head(teacher(teacher_x))

            loss = loss_fn(student_out, teacher_out)

            opt.zero_grad()
            loss.backward()
            opt.step()

            momentum = 1 - (1 - momentum_base) * (math.cos(epoch / epochs * math.pi) + 1) / 2
            ema_update(teacher, student, momentum)
            ema_update(teacher_head, student_head, momentum)

            epoch_loss += loss.item()
            current_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch_loss': f'{loss.item():.4f}'
            })


        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(loader):.4f}')
        avg_epoch_loss = epoch_loss / len(loader)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join('models', 'best_model.pth')
            save_checkpoint(student, teacher, student_head, teacher_head, opt, epoch + 1, avg_epoch_loss, best_path)
            print(f'New best model saved with loss: {best_loss:.4f}')


if __name__ == "__main__":
    main()