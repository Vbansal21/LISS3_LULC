import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import rasterio
from typing import Tuple, Optional

class LandsatAdapter:
    """Adapter class to convert RGB images to Landsat-like format."""
    
    def __init__(self, num_bands: int = 7):
        self.num_bands = num_bands
    
    def rgb_to_landsat(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to simulated Landsat bands.
        
        This is a simplified simulation. In reality, Landsat bands would have
        different spectral characteristics.
        
        Args:
            rgb_image: RGB image tensor of shape (3, H, W)
            
        Returns:
            Tensor of shape (num_bands, H, W) simulating Landsat bands
        """
        # Ensure input is correct shape
        assert rgb_image.shape[0] == 3, "Input must be RGB image"
        
        # Create empty tensor for all bands
        landsat_bands = torch.zeros((self.num_bands,) + rgb_image.shape[1:])
        
        # Copy RGB bands (assuming they correspond to Landsat RGB bands)
        landsat_bands[1:4] = rgb_image  # Bands 2-4 in Landsat 8
        
        # Simulate other bands using combinations of RGB
        # Band 1 (Coastal/Aerosol) - simulated as weighted sum of blue and green
        landsat_bands[0] = 0.7 * rgb_image[2] + 0.3 * rgb_image[1]
        
        # Band 5 (NIR) - simulated using red and green
        landsat_bands[4] = 0.8 * rgb_image[0] + 0.2 * rgb_image[1]
        
        # Band 6 (SWIR 1) - simulated using all bands
        landsat_bands[5] = 0.4 * rgb_image[0] + 0.3 * rgb_image[1] + 0.3 * rgb_image[2]
        
        # Band 7 (SWIR 2) - simulated using red and blue
        landsat_bands[6] = 0.6 * rgb_image[0] + 0.4 * rgb_image[2]
        
        return landsat_bands

def get_landsat_transforms(is_training: bool = True) -> transforms.Compose:
    """Get transforms for Landsat-like data."""
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if is_training:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ] + transform_list
    
    return transforms.Compose(transform_list)

class LandsatDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper class to convert RGB dataset to Landsat-like format."""
    
    def __init__(self, base_dataset: torch.utils.data.Dataset):
        self.base_dataset = base_dataset
        self.adapter = LandsatAdapter()
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.base_dataset[idx]
        
        # Convert RGB to Landsat-like format
        landsat_image = self.adapter.rgb_to_landsat(image)
        
        return landsat_image, label

def get_landsat_wrapped_dataloader(
    base_dataloader_fn,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Get a dataloader that provides Landsat-like data."""
    # Get base dataset
    base_dataset = base_dataloader_fn(split=split, batch_size=1, num_workers=0).dataset
    
    # Wrap dataset to provide Landsat-like data
    landsat_dataset = LandsatDatasetWrapper(base_dataset)
    
    # Create new dataloader
    dataloader = torch.utils.data.DataLoader(
        landsat_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        **kwargs
    )
    
    return dataloader 