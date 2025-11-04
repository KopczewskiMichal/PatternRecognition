import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from src.dataset import get_cifar10_dataloaders, get_mnist_dataloaders
from src.model import SimCLRViT, ViTWithClassifier
import wandb

#pretrain: python train.py --mode simclr --batch_size 32 --epochs 20 --num_workers 0 --wandb
# #fine-tune: python train.py --mode classifier --pretrained_simclr simclr_vit.pth --batch_size 128 --epochs 10 --num_workers 0 --wandb
def info_nce_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    pos = torch.cat([torch.arange(batch_size, 2 * batch_size),
                     torch.arange(0, batch_size)]).to(z.device)
    sim_pos = sim[torch.arange(2 * batch_size), pos]

    loss = -torch.log(
        torch.exp(sim_pos) / torch.exp(sim).sum(dim=1)
    ).mean()
    return loss


def train_simclr(model, train_loader, device, epochs, lr, use_wandb=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} starting...")
        running_loss = 0.0
        for batch_idx, (x_i, x_j) in enumerate(train_loader):
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            loss = info_nce_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"  Batch {batch_idx + 1}, Avg Loss={avg_loss:.4f}")
                if use_wandb:
                    wandb.log({"SimCLR Batch Loss": avg_loss})
                running_loss = 0.0
        if use_wandb:
            wandb.log({"SimCLR Epoch": epoch + 1, "SimCLR Epoch Loss": loss.item()})


def evaluate_classifier(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    acc = 100.0 * correct / total
    return acc


def train_classifier(model, train_loader, test_loader, device, epochs, lr, use_wandb=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate_classifier(model, test_loader, device)

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        if use_wandb:
            wandb.log({
                "Classifier Train Loss": avg_loss,
                "Classifier Train Acc": train_acc,
                "Classifier Test Acc": test_acc,
                "Epoch": epoch + 1
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=['simclr', 'classifier'], default='simclr')
    parser.add_argument("--pretrained_simclr", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        simclr=(args.mode == 'simclr')
    )

    if args.wandb:
        wandb.init(
            project="SimCLR-ViT",
            config={
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "mode": args.mode
            },
            name=f"{args.mode}_vit_run"
        )

    if args.mode == 'simclr':
        model = SimCLRViT(projection_dim=128)
        if args.pretrained_simclr:
            model.load_state_dict(torch.load(args.pretrained_simclr))
        train_simclr(model, train_loader, device, args.epochs, args.lr, use_wandb=args.wandb)
        torch.save(model.state_dict(), "simclr_vit.pth")

    else:  # classifier fine-tuning
        simclr_model = SimCLRViT(projection_dim=128)
        if args.pretrained_simclr:
            simclr_model.load_state_dict(torch.load(args.pretrained_simclr))
        model = ViTWithClassifier(simclr_model, num_classes=10)
        train_classifier(model, train_loader, test_loader, device, args.epochs, args.lr, use_wandb=args.wandb)
        torch.save(model.state_dict(), "vit_cifar10.pth")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
