from __future__ import print_function

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CameleyonBagDataset import CameleyonBagDataset
from model import Attention, GatedAttention

# Ustawienia treningu
parser = argparse.ArgumentParser(description='PyTorch CAMELEYON16 MIL Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--bag_size', type=int, default=100, metavar='BS',
                    help='number of patches per bag (WSI)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

# ŚCIEŻKI DO DANYCH (Zmień na swoje katalogi)
parser.add_argument('--normal_dir', type=str, default=r'D:\CAMELEYON16\training\normal', help='Path to normal .tif files')
parser.add_argument('--tumor_dir', type=str, default=r"D:\CAMELEYON16\training\tumor", help='Path to tumor .tif files')
parser.add_argument('--annot_dir', type=str, default=r"D:\CAMELEYON16\training\lesion_annotations", help='Path to annotation .xml files')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')
else:
    print('\nGPU is OF!')

print('Load Train Set...')
loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# Inicjalizacja datasetu
train_set = CameleyonBagDataset(
    normal_dir=args.normal_dir,
    tumor_dir=args.tumor_dir,
    annot_dir=args.annot_dir,
    bag_size=args.bag_size
)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, **loader_kwargs)

print(f'Train set size: {len(train_set)} bags (WSI slides)')

print('Init Model...')
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

    # Owijamy loader w tqdm dla paska postępu
    # desc - opis po lewej stronie paska
    # leave=False - czyści pasek po zakończeniu epoki (opcjonalne)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='bag')

    for batch_idx, (data, label) in enumerate(pbar):
        bag_label = label[0]

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        # Reset gradientów
        optimizer.zero_grad()

        # Obliczenia
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.item()

        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error

        # Backward pass
        loss.backward()
        optimizer.step()

        # Aktualizacja paska postępu o bieżący loss i błąd
        # To pozwala widzieć czy model "żyje" w trakcie trwania epoki
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Err': f'{error:.2f}'})

    # Obliczanie średnich po epoce
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    # Wypisanie podsumowania epoki (nadpisze ostatni stan paska lub wypisze pod nim)
    print(f'Epoch: {epoch} finished. Avg Loss: {train_loss:.4f}, Avg Error: {train_error:.4f}')


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)

        # Zapisywanie modelu co 5 epok
        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')