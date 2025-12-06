from __future__ import print_function

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from dataset_cameleyon import CameleyonBagDataset, CameleyonTestDataset
from model import Attention, GatedAttention
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser(description='PyTorch CAMELEYON16 MIL Example')
parser.add_argument('--bag_size', type=int, default=50, metavar='BS',
                    help='number of patches per bag (WSI)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

parser.add_argument('--train_normal_dir', type=str, default=r'D:\CAMELEYON16\training\normal',
                    help='Path to TRAIN normal .tif files')
parser.add_argument('--train_tumor_dir', type=str, default=r"D:\CAMELEYON16\training\tumor",
                    help='Path to TRAIN tumor .tif files')
parser.add_argument('--train_annot_dir', type=str, default=r"D:\CAMELEYON16\training\lesion_annotations",
                    help='Path to TRAIN annotation .xml files')

# --- ŚCIEŻKI TESTOWE (Bez XML) ---
parser.add_argument('--test_normal_dir', type=str, default=r"D:\CAMELEYON16\test\images\normal")
parser.add_argument('--test_tumor_dir', type=str, default=r"D:\CAMELEYON16\test\images\tumor")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print('\nGPU is ON! (Benchmark mode enabled)')
else:
    print('\nGPU is OFF!')

loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

print('\n--- LOADING TRAIN SET ---')
try:
    train_set = CameleyonBagDataset(
        normal_dir=args.train_normal_dir,
        tumor_dir=args.train_tumor_dir,
        annot_dir=args.train_annot_dir,
        bag_size=args.bag_size
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, **loader_kwargs)
    print(f'Train Set: {len(train_set)} slides')
except Exception as e:
    print(f"CRITICAL ERROR loading Train Set: {e}")
    sys.exit(1)

print('\n--- LOADING TEST SET ---')
try:
    test_set = CameleyonTestDataset(
        normal_dir=args.test_normal_dir,
        tumor_dir=args.test_tumor_dir,
        bag_size=args.bag_size
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_kwargs)
    print(f'Test Set: {len(test_set)} slides')
except Exception as e:
    print(f"CRITICAL ERROR loading Test Set: {e}")
    sys.exit(1)

print('\n--- INIT MODEL ---')
if args.model == 'attention':
    model = Attention()
elif args.model == 'gated_attention':
    model = GatedAttention()

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


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

            tqdm.write(f'[Test] Batch {batch_idx}: True: {int(bag_label.item())} | Pred: {int(predicted_label.item())}')

            # Wypisujemy aktualny loss w pasku
            pbar.set_postfix({'TLoss': f'{loss.item():.4f}'})

    test_loss /= len(loader)
    test_error /= len(loader)

    print(f'  Test Set -> Loss: {test_loss:.4f}, Error: {test_error:.4f}, Accuracy: {1.0 - test_error:.4f}')
    return test_loss, test_error


def save_random_bag_visualization(loader, filename='bag_visualization.png'):
    data_iter = iter(loader)
    bag, label = next(data_iter)

    bag = bag[0]
    label_val = int(label[0].item())

    num_samples = 10
    total_patches = bag.shape[0]
    indices = random.sample(range(total_patches), min(num_samples, total_patches))

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 3))
    label_text = "TUMOR (Rak)" if label_val == 1 else "NORMAL (Zdrowy)"
    fig.suptitle(f'Bag Label: {label_val} [{label_text}]', fontsize=16)

    for i, idx in enumerate(indices):
        img_tensor = bag[idx]  # [C, H, W]

        img = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Obsługa skali szarości (jeśli C=1)
        if img.shape[2] == 1:
            img = img.squeeze(2)  # [H, W]
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[i].imshow(img)

        axes[i].axis('off')
        axes[i].set_title(f'Idx: {idx}')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Zamknij, żeby nie wyświetlać w notebooku/oknie
    print(f"-> Zapisano podgląd baga do pliku: {filename}")


if __name__ == "__main__":
    save_random_bag_visualization(train_loader, filename='bag_visualization_train_level_x.png')
    save_random_bag_visualization(test_loader, filename='bag_visualization_test_level_x.png')
    #
    # print('\nStart Training loop...')
    # best_test_error = float('inf')
    #
    # for epoch in range(1, args.epochs + 1):
    #     t_loss, t_err = train(epoch)
    #     print(f'  Train Summary -> Loss: {t_loss:.4f}, Error: {t_err:.4f}')
    #
    #     test_loss, test_err = test(test_loader)
    #
    #     if test_err < best_test_error:
    #         print(f'  [SAVE] New best Test Error: {test_err:.4f} (was {best_test_error:.4f}). Saving model...')
    #         best_test_error = test_err
    #         torch.save(model.state_dict(), 'best_model.pth')
    #
