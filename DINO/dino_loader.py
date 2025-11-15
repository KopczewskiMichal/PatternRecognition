import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def make_transforms(image_size=128, local_size=96, local_views=6):
    global_t = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    local_t = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(local_size, scale=(0.1, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # wrapper top-level funkcja (pickle-friendly)
    def wrapper(img):
        global_imgs = [global_t(img), global_t(img)]
        local_imgs = [local_t(img) for _ in range(local_views)]
        return global_imgs + local_imgs

    return wrapper


def get_dino_loader(
        root="./../data",
        image_size=128,
        local_size=96,
        local_views=6,
        batch_size=64,
        shuffle=True,
        num_workers=0
):
    transform = make_transforms(image_size=image_size, local_size=local_size, local_views=local_views)
    dataset = FashionMNIST(root=root, download=True, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    return loader
