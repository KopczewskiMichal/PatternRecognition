import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2

def get_simclr_transform():
    return transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

def get_standard_transform():
    return transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])



def get_cifar10_dataloaders(batch_size=256, num_workers=4, simclr=True):
    base_train = datasets.CIFAR10(root='./data', train=True, download=True)
    base_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=get_standard_transform())

    if simclr:
        train_dataset = SimCLRDataset(base_train, get_simclr_transform())
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=get_standard_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        base_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    return train_loader, test_loader


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(batch_size=256, num_workers=4, simclr=True):
    # Define transforms
    def get_standard_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_simclr_transform():
        # Example simple augmentation for MNIST
        return transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Load datasets
    base_train = datasets.MNIST(root='./data', train=True, download=True)
    base_test = datasets.MNIST(root='./data', train=False, download=True, transform=get_standard_transform())

    # Prepare train dataset
    if simclr:
        train_dataset = SimCLRDataset(base_train, get_simclr_transform())  # reuse your SimCLRDataset class
    else:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=get_standard_transform())

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        base_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

