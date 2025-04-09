"""
UC Merced Land Use Dataset Loader

This module provides a PyTorch Dataset class for loading and processing the UC Merced Land Use Dataset.
The dataset contains aerial images of various land use categories.

Features:
- Efficient data loading with caching
- Support for distributed training
- Customizable transforms
- Automatic class balancing
- Support for both training and validation splits
- Memory-efficient processing
- Progress tracking with tqdm
- Comprehensive error handling

Author: Your Name
Date: 2024
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Union, Callable
from pathlib import Path
from tqdm import tqdm
import logging
from config import DATASET  # Import the config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCMercedDataset(Dataset):
    """
    UC Merced Land Use Dataset Loader
    
    This class implements a PyTorch Dataset for the UC Merced Land Use Dataset.
    It supports efficient data loading, caching, and distributed training.
    
    Args:
        root_dir (str): Path to the root directory containing the dataset
        transform (Optional[Callable]): Transform to apply to the images
        split (str): Either 'train' or 'val' to specify the dataset split
        val_split (float): Proportion of data to use for validation (default: 0.2)
        cache_images (bool): Whether to cache images in memory (default: False)
        distributed (bool): Whether to use distributed training (default: False)
        world_size (int): Number of processes in distributed training
        rank (int): Rank of current process in distributed training
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        split: str = 'train',
        val_split: float = 0.2,
        cache_images: bool = False,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42
    ):
        self.root_dir = os.path.join(root_dir, "Images")
        self.transform = transform
        self.split = split
        self.val_split = val_split
        self.cache_images = cache_images
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize image cache
        self.image_cache = {}
        
        # Load class information - ensure we only list directories
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        if not self.classes:
             raise FileNotFoundError(f"No class directories found in {self.root_dir}. Please check the path.")
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all samples
        self.samples = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_path in glob.glob(os.path.join(class_path, "*.tif")):
                self.samples.append((img_path, self.class_to_idx[cls]))
        
        # Split dataset
        self._split_dataset()
        
        # Log dataset statistics
        self._log_dataset_stats()
    
    def _split_dataset(self) -> None:
        """Split the dataset into training and validation sets"""
        # Calculate split indices
        num_samples = len(self.samples)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * (1 - self.val_split))
        
        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        # Adjust indices for distributed training
        if self.distributed:
            self.indices = self.indices[self.rank::self.world_size]
    
    def _log_dataset_stats(self) -> None:
        """Log dataset statistics"""
        logger.info(f"UC Merced Dataset ({self.split} split)")
        logger.info(f"Number of classes: {len(self.classes)}")
        logger.info(f"Number of samples: {len(self.indices)}")
        logger.info(f"Classes: {self.classes}")
        
        # Log class distribution
        class_counts = {cls: 0 for cls in self.classes}
        for idx in self.indices:
            _, label = self.samples[idx]
            class_counts[self.classes[label]] += 1
        
        logger.info("Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  {cls}: {count} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            Tuple[torch.Tensor, int, str]: Image tensor, label index, and class name
        """
        # Get actual index from split indices
        actual_idx = self.indices[idx]
        img_path, label = self.samples[actual_idx]
        
        # Try to get image from cache
        if self.cache_images and img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            # Load and cache image
            try:
                image = Image.open(img_path).convert('RGB')
                if self.cache_images:
                    self.image_cache[img_path] = image
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # Return a black image as fallback
                image = Image.new('RGB', (256, 256))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, self.classes[label]
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance
        
        Returns:
            torch.Tensor: Tensor of class weights
        """
        # Count samples per class
        class_counts = torch.zeros(len(self.classes))
        for idx in self.indices:
            _, label = self.samples[idx]
            class_counts[label] += 1
        
        # Calculate weights (inverse of frequency)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.classes)  # Normalize
        
        return weights

def get_ucmerced_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    split: str = 'train',
    val_split: float = 0.2,
    cache_images: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the UC Merced dataset
    
    Args:
        root_dir (str): Path to the root directory containing the dataset
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes for data loading
        transform (Optional[Callable]): Transform to apply to the images
        split (str): Either 'train' or 'val' to specify the dataset split
        val_split (float): Proportion of data to use for validation
        cache_images (bool): Whether to cache images in memory
        distributed (bool): Whether to use distributed training
        world_size (int): Number of processes in distributed training
        rank (int): Rank of current process in distributed training
        seed (int): Random seed for reproducibility
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        drop_last (bool): Whether to drop the last incomplete batch
        
    Returns:
        DataLoader: Configured DataLoader for the UC Merced dataset
    """
    # Create dataset
    dataset = UCMercedDataset(
        root_dir=root_dir,
        transform=transform,
        split=split,
        val_split=val_split,
        cache_images=cache_images,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=seed
    )
    
    # Create sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
    else:
        sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader

# Default transforms
def get_default_transforms(input_size: int = 224, mean: List[float] = None, std: List[float] = None, include_segmentation_augmentations: bool = False) -> Dict[str, transforms.Compose]:
    """
    Get default transforms for training and validation.
    
    Args:
        input_size: Size of input images
        mean: Mean values for normalization (if None, uses UCMerced defaults from config)
        std: Standard deviation values for normalization (if None, uses UCMerced defaults from config)
        include_segmentation_augmentations: Whether to include segmentation-specific augmentations
    
    Returns:
        Dictionary containing train and validation transforms
    """
    # Use UCMerced normalization parameters from config if not provided
    if mean is None:
        mean = DATASET['normalization']['ucmerced']['mean']
    if std is None:
        std = DATASET['normalization']['ucmerced']['std']
        
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train_transform': train_transform,
        'val_transform': val_transform
    }

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Test UC Merced Dataset Loader")
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='../data/UCMerced_LandUse/UCMerced_LandUse/',  # Default relative path
        help='Path to the UC Merced dataset directory'
    )
    args = parser.parse_args()

    print("="*30)
    print("RUNNING UC MERCED DATASET TESTS...")
    print(f"Using dataset path: {args.data_path}")
    print("="*30)

    # Check if path exists
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset path not found: {args.data_path}")
        logger.error("Please provide a valid path using --data_path argument.")
        exit(1)

    # --- 1. Test Default Transforms ---
    print("\n--- Testing get_default_transforms ---")
    transforms_dict = get_default_transforms()
    assert 'train_transform' in transforms_dict and isinstance(transforms_dict['train_transform'], transforms.Compose)
    assert 'val_transform' in transforms_dict and isinstance(transforms_dict['val_transform'], transforms.Compose)
    print("Default transforms created successfully.")
    print(f"Train Transform: {transforms_dict['train_transform']}")
    print(f"Val Transform: {transforms_dict['val_transform']}")

    # --- 2. Test Dataset Instantiation ---
    print("\n--- Testing UCMercedDataset Instantiation ---")
    try:
        print("Testing Train Split...")
        train_dataset = UCMercedDataset(
            root_dir=args.data_path,
            transform=transforms_dict['train_transform'],
            split='train'
        )
        print(f"Train dataset created with {len(train_dataset)} samples.")

        print("Testing Validation Split...")
        val_dataset = UCMercedDataset(
            root_dir=args.data_path,
            transform=transforms_dict['val_transform'],
            split='val'
        )
        print(f"Validation dataset created with {len(val_dataset)} samples.")
        
        print("Testing with Caching Enabled...")
        train_dataset_cached = UCMercedDataset(
            root_dir=args.data_path,
            transform=transforms_dict['train_transform'],
            split='train',
            cache_images=True
        )
        # Access a few items to populate cache
        _ = train_dataset_cached[0]
        _ = train_dataset_cached[1]
        print(f"Cached train dataset created. Cache size: {len(train_dataset_cached.image_cache)}")

        print("Testing Distributed Setting (Simulated Rank 0)...")
        dist_dataset_rank0 = UCMercedDataset(
            root_dir=args.data_path,
            transform=transforms_dict['train_transform'],
            split='train',
            distributed=True,
            world_size=2,
            rank=0
        )
        print(f"Distributed dataset (Rank 0) created with {len(dist_dataset_rank0)} samples.")

        print("Testing Distributed Setting (Simulated Rank 1)...")
        dist_dataset_rank1 = UCMercedDataset(
            root_dir=args.data_path,
            transform=transforms_dict['train_transform'],
            split='train',
            distributed=True,
            world_size=2,
            rank=1
        )
        print(f"Distributed dataset (Rank 1) created with {len(dist_dataset_rank1)} samples.")
        
        # Basic check for distributed split correctness
        assert len(dist_dataset_rank0) + len(dist_dataset_rank1) == len(train_dataset), \
               "Distributed dataset sizes do not sum up correctly."
        
        print("Dataset instantiation tests passed.")
        
    except Exception as e:
        logger.error(f"Error during dataset instantiation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 3. Test __getitem__ ---
    print("\n--- Testing __getitem__ ---")
    try:
        print("Loading sample 0 from train dataset...")
        img, label, label_name = train_dataset[0]
        print(f"Sample 0: Image Shape={img.shape}, Type={img.dtype}, Label={label}, Name={label_name}")
        assert isinstance(img, torch.Tensor) and len(img.shape) == 3, "Incorrect image format"
        assert isinstance(label, int), "Incorrect label format"
        assert isinstance(label_name, str) and label_name in train_dataset.classes, "Incorrect label name"

        print("Loading sample 0 from cached dataset...")
        img_cached, _, _ = train_dataset_cached[0]
        assert torch.equal(img, img_cached), "Cached image differs from non-cached"
        
        print("Loading sample 0 from validation dataset...")
        img_val, label_val, label_name_val = val_dataset[0]
        print(f"Sample 0 (Val): Image Shape={img_val.shape}, Type={img_val.dtype}, Label={label_val}, Name={label_name_val}")
        assert isinstance(img_val, torch.Tensor) and len(img_val.shape) == 3

        # Visualize the first sample
        plt.figure(figsize=(5, 5))
        # Need to unnormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img * std + mean 
        img_display = img_display.permute(1, 2, 0).numpy().clip(0, 1) # C,H,W -> H,W,C
        plt.imshow(img_display)
        plt.title(f"Sample 0: {label_name} (Label {label})")
        plt.axis('off')
        plt.show(block=False) # Use block=False to avoid stopping the script
        plt.pause(2) # Pause for 2 seconds to show the plot
        plt.close()
        print("Sample visualization displayed.")

        print("__getitem__ tests passed.")
    except Exception as e:
        logger.error(f"Error during __getitem__ test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 4. Test Class Weights ---
    print("\n--- Testing get_class_weights ---")
    try:
        weights = train_dataset.get_class_weights()
        print(f"Calculated class weights (Tensor shape: {weights.shape}):")
        print(weights)
        assert isinstance(weights, torch.Tensor) and weights.shape[0] == len(train_dataset.classes)
        print("Class weights calculation passed.")
    except Exception as e:
        logger.error(f"Error calculating class weights: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 5. Test Dataloader Creation ---
    print("\n--- Testing get_ucmerced_dataloader ---")
    try:
        print("Creating Train Dataloader (Batch Size=4, Workers=2)...")
        train_loader = get_ucmerced_dataloader(
            root_dir=args.data_path,
            batch_size=4,
            num_workers=2,
            transform=transforms_dict['train_transform'],
            split='train',
            pin_memory=True,
            drop_last=True
        )
        print("Train DataLoader created successfully.")

        print("Creating Validation Dataloader (Batch Size=8, Workers=0)...")
        val_loader = get_ucmerced_dataloader(
            root_dir=args.data_path,
            batch_size=8,
            num_workers=0,
            transform=transforms_dict['val_transform'],
            split='val',
            pin_memory=False,
            drop_last=False
        )
        print("Validation DataLoader created successfully.")
        
        print("Testing iteration through Train Dataloader...")
        train_batch = next(iter(train_loader))
        imgs, labels, names = train_batch
        print(f"Train Batch: Images Shape={imgs.shape}, Labels Shape={labels.shape}, Num Names={len(names)}")
        assert imgs.shape[0] == 4 and len(imgs.shape) == 4, "Incorrect train batch image shape"
        assert labels.shape[0] == 4, "Incorrect train batch label shape"

        print("Testing iteration through Validation Dataloader...")
        val_batch = next(iter(val_loader))
        imgs_val, labels_val, names_val = val_batch
        print(f"Val Batch: Images Shape={imgs_val.shape}, Labels Shape={labels_val.shape}, Num Names={len(names_val)}")
        # Batch size might be smaller than 8 if drop_last=False and dataset size is not multiple of 8
        assert imgs_val.shape[0] <= 8 and len(imgs_val.shape) == 4, "Incorrect val batch image shape"
        assert labels_val.shape[0] <= 8, "Incorrect val batch label shape"
        
        # Test distributed dataloader creation (simulated)
        print("Creating Distributed Train Dataloader (Simulated)...")
        dist_loader = get_ucmerced_dataloader(
            root_dir=args.data_path,
            batch_size=4,
            num_workers=0,
            transform=transforms_dict['train_transform'],
            split='train',
            distributed=True,
            world_size=2,
            rank=0
        )
        assert isinstance(dist_loader.sampler, DistributedSampler), "Distributed sampler not used"
        print("Distributed DataLoader created successfully.")

        print("Dataloader creation and iteration tests passed.")
    except Exception as e:
        logger.error(f"Error during dataloader test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "="*30)
    print("ALL UC MERCED DATASET TESTS PASSED SUCCESSFULLY!")
    print("="*30) 