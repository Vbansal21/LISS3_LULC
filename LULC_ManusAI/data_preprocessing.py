import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
from torchvision import transforms
import random

class UCMercedDataset(Dataset):
    """UC Merced Land Use Dataset with consistent image sizes"""
    def __init__(self, root_dir, transform=None, target_size=(128, 128)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Target size for all images and labels to ensure consistency.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.tif', '.jpg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize image to ensure consistent size
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Create a label image with the class index
        label_image = torch.zeros(self.target_size, dtype=torch.long)
        label_image.fill_(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_image

class ESAWorldCoverDataset(Dataset):
    """ESA World Cover Dataset with consistent patch sizes"""
    def __init__(self, tif_path, transform=None, patch_size=128, num_samples=100):
        """
        Args:
            tif_path (string): Path to the ESA World Cover GeoTIFF file.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size (int): Size of patches to extract.
            num_samples (int): Number of random samples to generate if tif_path doesn't exist.
        """
        self.tif_path = tif_path
        self.transform = transform
        self.patch_size = patch_size
        self.num_samples = num_samples
        
        # Check if the file exists
        if os.path.exists(tif_path):
            # Open the GeoTIFF file
            with rasterio.open(tif_path) as src:
                self.height = src.height
                self.width = src.width
                self.num_bands = src.count
                
                # Calculate number of patches
                self.num_patches_h = self.height // self.patch_size
                self.num_patches_w = self.width // self.patch_size
                self.num_patches = self.num_patches_h * self.num_patches_w
        else:
            # Create a placeholder dataset
            print(f"Warning: {tif_path} not found. Creating a placeholder dataset.")
            self.height = 1000
            self.width = 1000
            self.num_bands = 1
            self.num_patches = num_samples
    
    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        # Check if the file exists
        if os.path.exists(self.tif_path):
            # Calculate patch coordinates
            patch_h = idx // self.num_patches_w
            patch_w = idx % self.num_patches_w
            
            h_start = patch_h * self.patch_size
            w_start = patch_w * self.patch_size
            
            # Open the GeoTIFF file and read the patch
            with rasterio.open(self.tif_path) as src:
                # Read image data
                image = src.read(
                    window=((h_start, h_start + self.patch_size), 
                            (w_start, w_start + self.patch_size))
                )
                
                # Transpose to (H, W, C) format
                image = np.transpose(image, (1, 2, 0))
                
                # Convert to PIL Image
                if image.shape[2] == 1:  # If single band
                    image = Image.fromarray(image.squeeze(), mode='L')
                else:
                    image = Image.fromarray(image, mode='RGB')
                
                # Create label (for this placeholder, we'll just use the first band as label)
                label = src.read(
                    1,  # First band
                    window=((h_start, h_start + self.patch_size), 
                            (w_start, w_start + self.patch_size))
                )
                
                # Convert label to tensor
                label = torch.from_numpy(label).long()
        else:
            # Generate random data for placeholder
            # Generate random RGB data (3 channels)
            image_array = np.random.rand(self.patch_size, self.patch_size, 3).astype(np.float32)
            image = Image.fromarray((image_array * 255).astype(np.uint8), mode='RGB')
            
            # Generate random label (11 classes for ESA WorldCover)
            label = torch.randint(0, 11, (self.patch_size, self.patch_size), dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class LandsatDataset(Dataset):
    """Landsat Dataset for wildfire prediction with consistent patch sizes"""
    def __init__(self, tif_path, transform=None, patch_size=128, num_samples=100):
        """
        Args:
            tif_path (string): Path to the Landsat GeoTIFF file.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size (int): Size of patches to extract.
            num_samples (int): Number of random samples to generate.
        """
        self.tif_path = tif_path
        self.transform = transform
        self.patch_size = patch_size
        self.num_samples = num_samples
        
        # Check if the file exists
        if os.path.exists(tif_path):
            # Open the GeoTIFF file
            with rasterio.open(tif_path) as src:
                self.height = src.height
                self.width = src.width
                self.num_bands = src.count
        else:
            # Create a placeholder dataset
            print(f"Warning: {tif_path} not found. Creating a placeholder dataset.")
            self.height = 1000
            self.width = 1000
            self.num_bands = 4  # RGB + NBR
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Check if the file exists
        if os.path.exists(self.tif_path):
            # Generate random patch coordinates
            h_start = random.randint(0, self.height - self.patch_size)
            w_start = random.randint(0, self.width - self.patch_size)
            
            # Open the GeoTIFF file and read the patch
            with rasterio.open(self.tif_path) as src:
                # Read image data
                image = src.read(
                    window=((h_start, h_start + self.patch_size), 
                            (w_start, w_start + self.patch_size))
                )
                
                # Transpose to (C, H, W) format for PyTorch
                image = torch.from_numpy(image).float()
                
                # Create a binary mask for wildfire (0: no fire, 1: fire)
                # This is a placeholder - in a real implementation, we would use actual fire data
                label = torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)
                
                # Add some random fire regions
                num_fire_regions = random.randint(0, 3)
                for _ in range(num_fire_regions):
                    center_h = random.randint(0, self.patch_size - 1)
                    center_w = random.randint(0, self.patch_size - 1)
                    radius = random.randint(5, 20)
                    
                    for h in range(max(0, center_h - radius), min(self.patch_size, center_h + radius)):
                        for w in range(max(0, center_w - radius), min(self.patch_size, center_w + radius)):
                            if (h - center_h)**2 + (w - center_w)**2 <= radius**2:
                                label[h, w] = 1
        else:
            # Generate random data for placeholder
            # Generate random RGB+NBR data (4 channels)
            image = torch.rand(4, self.patch_size, self.patch_size, dtype=torch.float32)
            
            # Generate random binary mask for wildfire (0: no fire, 1: fire)
            label = torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)
            
            # Add some random fire regions
            num_fire_regions = random.randint(0, 3)
            for _ in range(num_fire_regions):
                center_h = random.randint(0, self.patch_size - 1)
                center_w = random.randint(0, self.patch_size - 1)
                radius = random.randint(5, 20)
                
                for h in range(max(0, center_h - radius), min(self.patch_size, center_h + radius)):
                    for w in range(max(0, center_w - radius), min(self.patch_size, center_w + radius)):
                        if (h - center_h)**2 + (w - center_w)**2 <= radius**2:
                            label[h, w] = 1
        
        if self.transform:
            # Apply transform if it's a PIL Image
            if not isinstance(image, torch.Tensor):
                image = self.transform(image)
            
        return image, label

def create_dataloaders(data_dir, task='landcover', batch_size=4, patch_size=128):
    """
    Create dataloaders for training, validation, and testing
    
    Args:
        data_dir (str): Path to data directory
        task (str): Task to create dataloaders for ('landcover' or 'wildfire')
        batch_size (int): Batch size for dataloaders
        patch_size (int): Size of patches to extract
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if task == 'landcover':
        # For land cover change prediction, we use UC Merced dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = UCMercedDataset(
            root_dir=os.path.join(data_dir, 'UCMerced_LandUse/UCMerced_LandUse/Images'),
            transform=transform,
            target_size=(patch_size, patch_size)
        )
        
        # Split into train, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
    elif task == 'wildfire':
        # For wildfire prediction, we use Landsat dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])  # 4 channels: RGB + NBR
        ])
        
        dataset = LandsatDataset(
            tif_path=os.path.join(data_dir, 'Landsat/LC09_NBR_20250114_182831_041036.tif'),
            transform=transform,
            patch_size=patch_size,
            num_samples=100
        )
        
        # Split into train, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def preprocess_ucmerced_dataset(data_dir, output_dir, patch_size=128):
    """Preprocess UC Merced dataset for land use classification"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = UCMercedDataset(
        root_dir=data_dir, 
        transform=transform,
        target_size=(patch_size, patch_size)
    )
    
    # Save class mapping
    class_mapping = {i: cls for i, cls in enumerate(dataset.classes)}
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for idx, class_name in class_mapping.items():
            f.write(f"{idx}: {class_name}\n")
    
    print(f"Preprocessed UC Merced dataset with {len(dataset)} samples")
    print(f"Class mapping saved to {os.path.join(output_dir, 'class_mapping.txt')}")
    
    return dataset

def preprocess_esa_worldcover_dataset(tif_path, output_dir, patch_size=128, num_samples=100):
    """Preprocess ESA World Cover dataset for land cover segmentation"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    dataset = ESAWorldCoverDataset(
        tif_path=tif_path, 
        transform=transform,
        patch_size=patch_size,
        num_samples=num_samples
    )
    
    # Save class mapping (ESA WorldCover has 11 classes)
    class_mapping = {
        0: "No data",
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen"
    }
    
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for idx, class_name in class_mapping.items():
            f.write(f"{idx}: {class_name}\n")
    
    print(f"Preprocessed ESA World Cover dataset with {len(dataset)} patches")
    print(f"Class mapping saved to {os.path.join(output_dir, 'class_mapping.txt')}")
    
    return dataset

def preprocess_landsat_dataset(tif_path, output_dir, patch_size=128, num_samples=100):
    """Preprocess Landsat dataset for wildfire prediction"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    dataset = LandsatDataset(
        tif_path=tif_path, 
        transform=transform, 
        patch_size=patch_size,
        num_samples=num_samples
    )
    
    # Save class mapping (binary: fire/no-fire)
    class_mapping = {
        0: "No fire",
        1: "Fire"
    }
    
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for idx, class_name in class_mapping.items():
            f.write(f"{idx}: {class_name}\n")
    
    print(f"Preprocessed Landsat dataset with {len(dataset)} patches")
    print(f"Class mapping saved to {os.path.join(output_dir, 'class_mapping.txt')}")
    
    return dataset