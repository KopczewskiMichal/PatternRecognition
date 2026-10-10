import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class MilPseudoBagDataset(Dataset):
    def __init__(self, split="train", bag_size=10, num_bags=1000, flatten_channels=True):
        self.bag_size = bag_size
        self.num_bags = num_bags
        self.flatten_channels = flatten_channels
        
        print(f"Loading dataset 'jxie/camelyon17' split '{split}'...")
        self.dataset = load_dataset("jxie/camelyon17", split=split)
        
        # Get all instance labels from the dataset
        all_instance_labels = np.array(self.dataset['label'])
        
        # 2. Pre-index indices by label for efficient sampling
        print("Indexing patches by label...")
        self.neg_indices = np.where(all_instance_labels == 0)[0]
        self.pos_indices = np.where(all_instance_labels == 1)[0]
        
        if len(self.pos_indices) == 0:
            raise ValueError("Split contains no positive instances. Cannot create positive bags.")

        # Store all instance labels globally for quick lookup later
        self.all_instance_labels_tensor = torch.tensor(all_instance_labels, dtype=torch.float32)

        # 3. Pre-generate bag indices and labels
        self.bag_indices = []
        self.bag_labels = [] # Stores Bag-level label (0 or 1)
        self.bag_instance_labels = [] # Stores Instance-level labels (Tensor of size N)
        self._create_balanced_bags()
        
        # Define transform for flattening/normalization
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1) if flatten_channels else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def _create_balanced_bags(self):
        """
        Generates indices for pseudo-bags with a 50/50 split.
        """
        num_pos_bags = self.num_bags // 2
        num_neg_bags = self.num_bags - num_pos_bags
        
        # --- Create Negative Bags (All instances must be label 0) ---
        for _ in range(num_neg_bags):
            indices = np.random.choice(self.neg_indices, size=self.bag_size, replace=True)
            
            # Retrieve instance labels
            instance_labels = self.all_instance_labels_tensor[indices]
            
            self.bag_indices.append(indices)
            self.bag_labels.append(0) # Bag label is 0
            self.bag_instance_labels.append(instance_labels)
            
        # --- Create Positive Bags (At least one instance must be label 1) ---
        for _ in range(num_pos_bags):
            n_pos_in_bag = np.random.randint(1, self.bag_size + 1) 
            n_neg_in_bag = self.bag_size - n_pos_in_bag
            
            pos_selection = np.random.choice(self.pos_indices, size=n_pos_in_bag, replace=True)
            neg_selection = np.random.choice(self.neg_indices, size=n_neg_in_bag, replace=True)
            
            indices = np.concatenate([pos_selection, neg_selection])
            np.random.shuffle(indices)
            
            # Retrieve instance labels
            instance_labels = self.all_instance_labels_tensor[indices]
            
            self.bag_indices.append(indices)
            self.bag_labels.append(1) # Bag label is 1 (guaranteed by selection)
            self.bag_instance_labels.append(instance_labels)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        """
        Returns:
            bag_images (Tensor): shape (bag_size, C, H, W)
            labels (list): [bag_label (scalar/max), instance_labels (Tensor)]
        """
        indices = self.bag_indices[idx]
        bag_label = self.bag_labels[idx] 
        instance_labels = self.bag_instance_labels[idx] 
        
        images_list = []
        for i in indices:
            img = self.dataset[int(i)]['image']
            if self.transform:
                img = self.transform(img)
            images_list.append(img)
            
        bag_tensor = torch.stack(images_list)
        
        labels_list = [
            bag_label,      
            instance_labels    
        ]

        
        return bag_tensor, labels_list