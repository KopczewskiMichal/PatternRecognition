import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from models import create_backbones, DinoHead, ema_update


device = "cuda"
image_size = 32
local_size = 16
local_crops = 6
batch_size = 64
epochs = 30
momentum_base = 0.996
lr = 1e-4
weight_decay = 1e-5

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_pretrained_backbone(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    student, teacher = create_backbones(image_size)

    student.load_state_dict(checkpoint['student_state_dict'])
    return student

def train_linear(backbone, classifier, train_loader, val_loader, save_path:str):
    backbone.eval()  # ZAMRAŻAMY backbone!
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(50):  # Krótki trening
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Linear Epoch {epoch + 1}/50')

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward - backbone zamrożony!
            with torch.no_grad():
                features = backbone(images)  # [B, feature_dim]

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / (total + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        val_acc = evaluate_linear(backbone, classifier, val_loader)
        print(f'Epoch {epoch + 1}: Val Acc = {val_acc:.2f}%')
        if correct/total > best_acc:
            torch.save(classifier.state_dict(), save_path)
            print("Best model saved")

def evaluate_linear(backbone, classifier, loader):
    backbone.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            features = backbone(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total

def main():
    backbone = load_pretrained_backbone('./models/best_model.pth')
    backbone = backbone.to(device)

    feature_dim = backbone.num_features
    classifier = LinearClassifier(feature_dim, num_classes=10).to(device)

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dataset = FashionMNIST(root="./../data", train=True, download=True, transform=train_transform)
    val_dataset = FashionMNIST(root="./../data", train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    train_linear(backbone, classifier, train_loader, val_loader, save_path='./models/best_linear_classifier.pth')

    final_acc = evaluate_linear(backbone, classifier, val_loader)
    print(f'Final Test Accuracy: {final_acc:.2f}%')

if __name__ == "__main__":
    main()