from __future__ import print_function

import argparse
import os.path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

from dataset_cameleyon import WholeSlideBagDataset
from model import Attention, GatedAttention, DINOAttention


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', unit='bag')

    for batch_idx, (data, label) in enumerate(pbar):
        bag_label = label[0]

        if args.cuda:
            data, bag_label = data.cuda(non_blocking=True), bag_label.cuda(non_blocking=True)

        optimizer.zero_grad()

        # Loss & Error
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error

        loss.backward()
        optimizer.step()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Err': f'{error:.2f}'})

    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    return train_loss, train_error


def test(loader):
    model.eval()
    test_loss = 0.
    test_error = 0.

    with torch.no_grad():
        pbar = tqdm(loader, desc='        [Test]', unit='bag')

        for batch_idx, (data, label) in enumerate(pbar):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(non_blocking=True), bag_label.cuda(non_blocking=True)

            loss, _ = model.calculate_objective(data, bag_label)
            test_loss += loss.item()
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            # tqdm.write(f'[Test] Batch {batch_idx}: True: {int(bag_label.item())} | Pred: {int(predicted_label.item())}')

            pbar.set_postfix({'TLoss': f'{loss.item():.4f}'})

    test_loss /= len(loader)
    test_error /= len(loader)
    print(f'  Test Set -> Loss: {test_loss:.4f}, Error: {test_error:.4f}, Accuracy: {1.0 - test_error:.4f}')
    return test_loss, test_error

def plot_and_save_history(history: list[tuple[float, float, float, float]], program_args,
                          filename: str = 'training_plots/training_history.png'):
    train_loss = [h[0] for h in history]
    train_acc = [h[1] for h in history]
    val_loss = [h[2] for h in history]
    val_acc = [h[3] for h in history]
    epochs = range(1, len(history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.plot(epochs, train_loss, 'r-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'b-', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    info_text = f"Parameters: {program_args}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=8, style='italic', alpha=0.8)


    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(filename)
    plt.close(fig)

    print(f"Plot saved: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CAMELEYON16 MIL Example')
    parser.add_argument('--bag_size', type=int, default=100, metavar='BS',
                        help='number of patches per bag (WSI)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--patch_size', type=int, default=96, metavar='N',
                        help='patch size')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-3, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
    parser.add_argument('--unfreeze_dino_blocks', type=int, default=2, help='Number of unfreezed DINO blocks at the end. Used only with dino encoder.')
    parser.add_argument('--data_dir', type=str, default=r'D:\CAMELEYON16\preprocessed',
                        help='Path to training and test files.')


    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    training_dir = os.path.join(args.data_dir, 'training')
    test_dir = os.path.join(args.data_dir, 'test')


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        print('\nGPU is ON')
    else:
        print('\nGPU is OFF')

    print('\n--- LOADING TRAIN SET ---')
    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
    ])

    try:
        train_ds = WholeSlideBagDataset(
            data_dir=training_dir,
            transform=train_transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=3
        )

        print(f'Train Set: {len(train_ds)} slides')
    except Exception as e:
        print(f"CRITICAL ERROR loading Train Set: {e}")
        sys.exit(1)

    print('\n--- LOADING TEST SET ---')
    try:
        test_ds = WholeSlideBagDataset(
            data_dir=test_dir,
            transform=train_transform
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            # prefetch_factor=2
        )

        print(f'Test Set: {len(test_ds)} slides')
    except Exception as e:
        print(f"CRITICAL ERROR loading Test Set: {e}")
        sys.exit(1)

    print('\n--- INIT MODEL ---')
    if args.model == 'attention':
        model = Attention(args.patch_size)
    elif args.model == 'gated_attention':
        model = GatedAttention(args.patch_size)

    match args.model:
        case 'attention':
            model = Attention(args.patch_size)
        case 'gated_attention':
            model = GatedAttention(args.patch_size)
        case 'dino_attention':
            model = DINOAttention(args.unfreeze_dino_blocks)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    print('\nStart Training loop...')
    best_test_error = float('inf')

    history = []
    for epoch in range(1, args.epochs + 1):
        t_loss, t_err = train(epoch)
        print(f'  Train Summary -> Loss: {t_loss:.4f}, Error: {t_err:.4f}')

        test_loss, test_err = test(test_loader)

        history.append((t_loss, 1.0 - t_err, test_loss, 1.0 - test_err))
        plot_and_save_history(history, args)

        if test_err < best_test_error:
            print(f'  [SAVE] New best Test Error: {test_err:.4f} (was {best_test_error:.4f}). Saving model...')
            best_test_error = test_err
            torch.save(model.state_dict(), 'best_model.pth')

