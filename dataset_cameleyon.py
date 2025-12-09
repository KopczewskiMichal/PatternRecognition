import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PreprocessedBagDataset(Dataset):
    def __init__(self, root_dir, bag_size=100, bags_per_slide=10,
                 tumor_ratio=0.5, transform=None, is_train=True):
        self.bag_size = bag_size
        self.bags_per_slide = bags_per_slide
        self.tumor_ratio = tumor_ratio
        self.is_train = is_train
        self.transform = transform
        normal_root_dir = os.path.join(root_dir, 'normal')
        print(normal_root_dir)
        tumor_root_dir = os.path.join(root_dir, 'tumor')


        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.data_list = []

        if normal_root_dir and os.path.exists(normal_root_dir):
            slides = [os.path.join(normal_root_dir, d) for d in os.listdir(normal_root_dir)
                      if os.path.isdir(os.path.join(normal_root_dir, d))]
            for s in slides:
                self.data_list.append({'path': s, 'label': 0})

        if tumor_root_dir and os.path.exists(tumor_root_dir):
            slides = [os.path.join(tumor_root_dir, d) for d in os.listdir(tumor_root_dir)
                      if os.path.isdir(os.path.join(tumor_root_dir, d))]
            for s in slides:
                self.data_list.append({'path': s, 'label': 1})

    def __len__(self):
        return len(self.data_list) * self.bags_per_slide

    def __getitem__(self, idx):
        slide_idx = idx // self.bags_per_slide
        slide_data = self.data_list[slide_idx]

        slide_path = slide_data['path']
        label = slide_data['label']

        path_n_sub = os.path.join(slide_path, 'normal')
        path_t_sub = os.path.join(slide_path, 'tumor')

        patches_paths = []

        if self.is_train:
            files_normal = [os.path.join(path_n_sub, f) for f in os.listdir(path_n_sub) if
                            f.endswith('.png')] if os.path.exists(path_n_sub) else []
            files_tumor = [os.path.join(path_t_sub, f) for f in os.listdir(path_t_sub) if
                           f.endswith('.png')] if os.path.exists(path_t_sub) else []

            if label == 1:
                n_tumor = int(self.bag_size * self.tumor_ratio)
                n_normal = self.bag_size - n_tumor

                p_t = random.choices(files_tumor, k=n_tumor) if files_tumor else []
                fallback_pool = files_normal if files_normal else files_tumor

                if not p_t and not fallback_pool:
                    return self._empty_bag(), torch.tensor([1.])

                p_n = random.choices(files_normal, k=n_normal) if files_normal else random.choices(fallback_pool,
                                                                                                   k=n_normal)
                patches_paths = p_t + p_n
            else:
                if not files_normal:
                    return self._empty_bag(), torch.tensor([0.])
                patches_paths = random.choices(files_normal, k=self.bag_size)

        else:
            if os.path.exists(path_n_sub):
                files_all = [os.path.join(path_n_sub, f) for f in os.listdir(path_n_sub) if f.endswith('.png')]

                if not files_all:
                    return self._empty_bag(), torch.tensor([float(label)])

                if len(files_all) < self.bag_size:
                    patches_paths = random.choices(files_all, k=self.bag_size)
                else:
                    patches_paths = random.sample(files_all, k=self.bag_size)
            else:
                return self._empty_bag(), torch.tensor([float(label)])

        while len(patches_paths) < self.bag_size:
            if not patches_paths:
                return self._empty_bag(), torch.tensor([float(label)])
            patches_paths.append(random.choice(patches_paths))

        random.shuffle(patches_paths)

        tensors = []
        for p in patches_paths:
            img = Image.open(p).convert('L')
            if self.transform:
                img = self.transform(img)
            tensors.append(img)

        bag_tensor = torch.stack(tensors)
        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    def _empty_bag(self):
        return torch.zeros(self.bag_size, 1, 28, 28)