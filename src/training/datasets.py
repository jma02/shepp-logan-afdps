"""PyTorch dataset utilities for training and evaluation."""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class PhantomsDataset(Dataset):
    """Custom dataset for loading phantoms.pt file with uniform dequantization."""
    
    def __init__(self, data_path, image_size=128, split='train', uniform_dequantization=True):
        """
        Args:
            data_path: Path to phantoms.pt file
            image_size: Size to resize images to (not used if already correct size)
            split: 'train', 'val', or 'test'
            uniform_dequantization: If True, apply uniform dequantization
        """
        # Load the phantoms data
        self.images = torch.load(data_path)  # Expected shape: [10000, 128, 128]
        
        # Add channel dimension if needed (from [N, H, W] to [N, C, H, W])
        if self.images.dim() == 3:
            self.images = self.images.unsqueeze(1)  # Add channel dimension
        # Split into train/val/test (80/10/10)
        n = len(self.images)
        train_size = int(0.8 * n)
        val_size = (n - train_size) // 2
        
        if split == 'train':
            self.images = self.images[:train_size]
        elif split == 'val':
            self.images = self.images[train_size:train_size + val_size]
        else:  # test
            self.images = self.images[train_size + val_size:]
        
        self.uniform_dequantization = uniform_dequantization
        self.image_size = image_size
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].float()  # Ensure float32
        
        # Apply uniform dequantization if needed
        if self.uniform_dequantization:
            img = (img + torch.rand_like(img)) / 256.0
        
        return {'image': img}


def get_dataloaders(config):
    """Create train and validation data loaders for the phantoms dataset."""
    phantoms_path = 'data/phantoms.pt'
    
    # Create datasets
    train_dataset = PhantomsDataset(
        data_path=phantoms_path,
        image_size=config.data.image_size,
        split='train',
        uniform_dequantization=config.data.uniform_dequantization
    )
    eval_dataset = PhantomsDataset(
        data_path=phantoms_path,
        image_size=config.data.image_size,
        split='val',
        uniform_dequantization=config.data.uniform_dequantization
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, eval_loader
