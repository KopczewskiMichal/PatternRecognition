import os
import glob
import torch
from PIL import Image
from numpy.f2py.auxfuncs import throw_error
from torch.nn.init import normal_
from torch.utils.data import Dataset
from torchvision import transforms


class WholeSlideBagDataset(Dataset):
    def __init__(self, data_dir:str, transform=None):
        normal_dir = os.path.join(data_dir, 'normal')
        tumor_dir = os.path.join(data_dir, 'tumor')
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.data_list = []

        if normal_dir and os.path.exists(normal_dir):
            slides = [os.path.join(normal_dir, d) for d in os.listdir(normal_dir)
                      if os.path.isdir(os.path.join(normal_dir, d))]
            for s in slides:
                self.data_list.append({'path': s, 'label': 0})

        if tumor_dir and os.path.exists(tumor_dir):
            slides = [os.path.join(tumor_dir, d) for d in os.listdir(tumor_dir)
                      if os.path.isdir(os.path.join(tumor_dir, d))]
            for s in slides:
                self.data_list.append({'path': s, 'label': 1})

        print(f"Dataset loaded: {len(self.data_list)} slides.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        slide_data = self.data_list[idx]
        slide_path = slide_data['path']
        label = slide_data['label']

        image_files = glob.glob(os.path.join(slide_path, "*.jpg"))

        # image_files.sort()

        if not image_files:
            raise Exception(f"No patches found in {slide_path}")
            # empty = torch.zeros(1, 1, 256, 256)
            # return empty, torch.tensor([float(label)])

        tensors = []
        for img_path in image_files:
            img = Image.open(img_path).convert('L')  # grayscale
            if img.size[0] != 96 or img.size[1] != 96:
                raise Exception(f'Incorrect size image {img_path}')

            if self.transform:
                img = self.transform(img)
            tensors.append(img)

        # (patches_count, 1, H, W)
        bag_tensor = torch.stack(tensors)

        return bag_tensor, torch.tensor([float(label)])