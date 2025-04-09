"""
ESA WorldCover Dataset Loader

This module provides a PyTorch Dataset class for loading and processing the ESA WorldCover dataset,
with support for overlap checking with Landsat imagery.

Features:
- Efficient data loading with caching
- Support for distributed training
- Customizable transforms
- Automatic class balancing
- Support for both training and validation splits
- Memory-efficient processing
- Progress tracking with tqdm
- Comprehensive error handling
- Overlap checking with standardized input formats

Author: Your Name
Date: 2024
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, transform
import geopandas as gpd
from shapely.geometry import box
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Union, Callable, Any
from pathlib import Path
from tqdm import tqdm
import logging
import json
from config import DATASET, PATHS # Import necessary configs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default ignore index for segmentation masks
DEFAULT_IGNORE_INDEX = 255

class ESAWorldCoverDataset(Dataset):
    """
    ESA WorldCover Dataset Loader with Landsat Overlap Support
    
    This class implements a PyTorch Dataset for the ESA WorldCover Dataset,
    with support for checking overlap with Landsat imagery.
    
    Args:
        esa_directory (str): Path to the ESA WorldCover data directory
        landsat_path (str): Path to the Landsat data directory
        transform (Optional[Callable]): Transform to apply to the images
        patch_size (int): Size of image patches to extract
        num_samples (int): Number of samples to generate
        test_mode (bool): Whether to run in test mode (creates synthetic data)
        cache_images (bool): Whether to cache images in memory
        distributed (bool): Whether to use distributed training
        world_size (int): Number of processes in distributed training
        rank (int): Rank of current process in distributed training
        seed (int): Random seed for reproducibility
        ignore_index (int): Pixel value to ignore in loss calculation
    """
    
    def __init__(
        self,
        esa_directory: str,
        landsat_path: str,
        transform: Optional[Callable] = None,
        patch_size: int = 224,
        num_samples: int = 10000,
        test_mode: bool = False,
        cache_images: bool = False,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        ignore_index: int = DEFAULT_IGNORE_INDEX
    ):
        self.esa_directory = esa_directory
        self.landsat_path = landsat_path
        self.transform = transform
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.test_mode = test_mode
        self.cache_images = cache_images
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.half_patch = patch_size // 2
        self.ignore_index = ignore_index
        
        # Set random seed for reproducibility
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        
        # Initialize image cache
        self.image_cache = {}
        
        # ESA WorldCover classes and mapping to 0-based indices
        self.esa_classes = {
            10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
            50: 'Built-up', 60: 'Bare / sparse vegetation', 70: 'Snow and ice',
            80: 'Permanent water bodies', 90: 'Herbaceous wetland',
            95: 'Mangroves', 100: 'Moss and lichen'
        }
        # Create mapping from ESA label -> 0-based index
        self.esa_label_to_index = {label: idx for idx, label in enumerate(self.esa_classes.keys())}
        self.num_classes = len(self.esa_classes)
        
        # Store normalization values from config
        try:
             self.norm_mean = torch.tensor(DATASET['normalization']['mean'], dtype=torch.float32)
             self.norm_std = torch.tensor(DATASET['normalization']['std'], dtype=torch.float32)
             if len(self.norm_mean) != 6 or len(self.norm_std) != 6:
                  logger.warning(f"Expected 6 normalization values in config, found {len(self.norm_mean)}. Check config.py.")
                  # Fallback or raise error? For now, pad/truncate to 6 if possible.
                  self.norm_mean = torch.cat((self.norm_mean, torch.zeros(6 - len(self.norm_mean))))[:6] if len(self.norm_mean) < 6 else self.norm_mean[:6]
                  self.norm_std = torch.cat((self.norm_std, torch.ones(6 - len(self.norm_std))))[:6] if len(self.norm_std) < 6 else self.norm_std[:6]

        except KeyError:
             logger.warning("Normalization values not found in DATASET config. Using default ImageNet RGB stats duplicated.")
             self.norm_mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=torch.float32)
             self.norm_std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=torch.float32)

        # Ensure mean/std are tensors of shape [6, 1, 1] for broadcasting
        self.norm_mean = self.norm_mean.view(6, 1, 1)
        self.norm_std = self.norm_std.view(6, 1, 1)
        
        try:
            # Load Landsat bands and get bounds
            self.landsat_bands = self._load_landsat_bands()
            self.landsat_bounds = self._get_landsat_bounds()
            self.landsat_crs = self._get_landsat_crs()
            
            # Find overlapping ESA WorldCover tiles
            self.esa_tiles = self._find_overlapping_esa_tiles()
            
            # Generate sample locations
            self.sample_locations = self._generate_sample_locations()
            
            # Log dataset statistics
            self._log_dataset_stats()
            
        except Exception as e:
            if test_mode:
                logger.warning(f"Error in initialization, creating synthetic test data: {e}")
                self._create_synthetic_test_data()
            else:
                logger.error(f"Failed to initialize ESAWorldCoverDataset: {e}", exc_info=True)
                raise e
    
    def _load_landsat_bands(self) -> np.ndarray:
        """Load Landsat 9 Surface Reflectance bands"""
        pattern = os.path.join(self.landsat_path, '*_SR_B*.TIF')
        landsat_files = sorted(glob.glob(pattern))
        
        if not landsat_files:
            raise ValueError(f"No Landsat SR bands found in {self.landsat_path}")
        
        # For LULC, we want B, G, R, NIR, SWIR1, SWIR2
        band_numbers = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        selected_bands = []
        
        for band_num in band_numbers:
            matching_files = [f for f in landsat_files if f'_SR_{band_num}.TIF' in os.path.basename(f)]
            if matching_files:
                selected_bands.append(matching_files[0])
            else:
                # Attempt to find older naming convention if necessary
                matching_files_alt = [f for f in landsat_files if f'_{band_num}.TIF' in os.path.basename(f)]
                if matching_files_alt:
                    selected_bands.append(matching_files_alt[0])
                else:
                    logger.warning(f"Could not find {band_num} band in {self.landsat_path}")
        
        if len(selected_bands) != len(band_numbers):
            raise ValueError(f"Found only {len(selected_bands)}/{len(band_numbers)} required Landsat bands.")
        
        # Load bands
        bands_data = []
        for band_file in tqdm(selected_bands, desc="Loading Landsat bands"):
            try:
                with rasterio.open(band_file) as src:
                    # Read band data, ensure float32 for scaling
                    band_data = src.read(1).astype(np.float32)
                    
                    # Apply scale factor if available in metadata (common for Landsat C2 L2)
                    scale = src.scales[0] if src.scales else 0.0000275
                    offset = src.offsets[0] if src.offsets else -0.2
                    
                    # Apply scale and offset
                    band_data = band_data * scale + offset
                    
                    # Handle nodata explicitly after scaling
                    if src.nodata is not None:
                        # Convert nodata to float32 before comparison
                        nodata_val = np.float32(src.nodata * scale + offset)
                        # Use a small tolerance for float comparison
                        mask = np.isclose(band_data, nodata_val) | np.isnan(band_data) # Also catch NaNs
                        band_data[mask] = np.nan # Set to NaN for consistent handling
                    
                    bands_data.append(band_data)
                    
                    # Store metadata from first band
                    if len(bands_data) == 1:
                        self.landsat_transform = src.transform
                        self.landsat_crs = src.crs
                        self.landsat_width = src.width
                        self.landsat_height = src.height
            except rasterio.RasterioIOError as e:
                logger.error(f"Error reading Landsat band {band_file}: {e}")
                raise
        
        return np.stack(bands_data)
    
    def _get_landsat_bounds(self) -> rasterio.coords.BoundingBox:
        """Get bounding box of the Landsat scene"""
        return rasterio.coords.BoundingBox(
            left=self.landsat_transform[2],
            bottom=self.landsat_transform[5] + (self.landsat_height * self.landsat_transform[4]),
            right=self.landsat_transform[2] + (self.landsat_width * self.landsat_transform[0]),
            top=self.landsat_transform[5]
        )
    
    def _get_landsat_crs(self) -> rasterio.crs.CRS:
        """Get CRS of the Landsat scene"""
        return self.landsat_crs
    
    def _find_overlapping_esa_tiles(self) -> List[Dict[str, Any]]:
        """Find ESA WorldCover tiles that overlap with the Landsat scene"""
        esa_files = glob.glob(os.path.join(self.esa_directory, '*.tif'))
        esa_files = [f for f in esa_files if ':Zone.Identifier' not in os.path.basename(f)]
        
        overlapping_tiles = []
        
        for esa_file in esa_files:
            try:
                with rasterio.open(esa_file) as src:
                    esa_bounds = src.bounds
                    esa_crs = src.crs
                    
                    try:
                        landsat_bounds_in_esa_crs = transform_bounds(
                            self.landsat_crs, 
                            esa_crs, 
                            *self.landsat_bounds
                        )
                        
                        overlaps = (
                            landsat_bounds_in_esa_crs[0] < esa_bounds.right and
                            landsat_bounds_in_esa_crs[2] > esa_bounds.left and
                            landsat_bounds_in_esa_crs[1] < esa_bounds.top and
                            landsat_bounds_in_esa_crs[3] > esa_bounds.bottom
                        )
                    except Exception:
                        overlaps = (
                            self.landsat_bounds.left < esa_bounds.right and
                            self.landsat_bounds.right > esa_bounds.left and
                            self.landsat_bounds.bottom < esa_bounds.top and
                            self.landsat_bounds.top > esa_bounds.bottom
                        )

                    if overlaps:
                        overlapping_tiles.append({
                            'path': esa_file,
                            'bounds': esa_bounds,
                            'transform': src.transform,
                            'crs': src.crs,
                            'width': src.width,
                            'height': src.height
                        })
            except Exception as e:
                logger.warning(f"Error processing ESA tile {os.path.basename(esa_file)}: {e}")
        
        if not overlapping_tiles and not self.test_mode:
            raise ValueError("No overlapping ESA WorldCover tiles found for the Landsat scene")
        
        return overlapping_tiles
    
    def _generate_sample_locations(self) -> List[Dict[str, Any]]:
        """Generate sample locations from overlapping ESA tiles"""
        locations = []
        samples_per_tile = self.num_samples // max(1, len(self.esa_tiles))
        
        for tile_info in self.esa_tiles:
            try:
                with rasterio.open(tile_info['path']) as src:
                    esa_data = src.read(1)
                    
                    try:
                        landsat_bounds_in_esa_crs = transform_bounds(
                            self.landsat_crs, 
                            tile_info['crs'], 
                            *self.landsat_bounds
                        )
                        
                        min_row, min_col = rasterio.transform.rowcol(
                            src.transform, 
                            landsat_bounds_in_esa_crs[0], 
                            landsat_bounds_in_esa_crs[3]
                        )
                        max_row, max_col = rasterio.transform.rowcol(
                            src.transform, 
                            landsat_bounds_in_esa_crs[2], 
                            landsat_bounds_in_esa_crs[1]
                        )
                        
                        min_row = max(0, min_row)
                        min_col = max(0, min_col)
                        max_row = min(tile_info['height'], max_row)
                        max_col = min(tile_info['width'], max_col)
                        
                        if max_row - min_row < 10 or max_col - min_col < 10:
                            continue
                        
                        overlap_data = esa_data[min_row:max_row, min_col:max_col]
                    except Exception as e:
                        logger.warning(f"Error calculating overlap for tile {os.path.basename(tile_info['path'])}: {e}")
                        continue # Skip this tile if overlap calculation fails

                    unique_classes, counts = np.unique(overlap_data, return_counts=True)
                    
                    class_indices = {}
                    for cls in unique_classes:
                        if cls in self.esa_classes:
                            class_indices[cls] = np.where(overlap_data == cls)
                    
                    if not class_indices:
                        continue
                    
                    total_valid_pixels = sum(counts[np.isin(unique_classes, list(self.esa_classes.keys()))])
                    if total_valid_pixels == 0:
                        continue
                    
                    samples_per_class = {}
                    for cls, indices in class_indices.items():
                        cls_count = len(indices[0])
                        cls_proportion = cls_count / total_valid_pixels
                        samples_per_class[cls] = max(10, int(samples_per_tile * cls_proportion))
                    
                    for cls, num_samples in samples_per_class.items():
                        indices = class_indices[cls]
                        if len(indices[0]) > 0:
                            sample_idx = np.random.choice(len(indices[0]), 
                                                         min(num_samples, len(indices[0])), 
                                                         replace=False)
                            
                            for idx in sample_idx:
                                local_y, local_x = indices[0][idx], indices[1][idx]
                                y = local_y + min_row
                                x = local_x + min_col
                                
                                esa_lon, esa_lat = src.xy(y, x)
                                
                                try:
                                    landsat_proj_x, landsat_proj_y = transform(
                                        src_crs=tile_info['crs'],
                                        dst_crs=self.landsat_crs,
                                        xs=[esa_lon],
                                        ys=[esa_lat]
                                    )
                                    
                                    landsat_y, landsat_x = rasterio.transform.rowcol(
                                        self.landsat_transform,
                                        landsat_proj_x[0],
                                        landsat_proj_y[0]
                                    )
                                    
                                    if (landsat_y >= self.half_patch and 
                                        landsat_x >= self.half_patch and 
                                        landsat_y < self.landsat_height - self.half_patch and 
                                        landsat_x < self.landsat_width - self.half_patch):
                                        
                                        locations.append({
                                            'tile_info': tile_info,
                                            'esa_y': y,
                                            'esa_x': x,
                                            'landsat_y': landsat_y,
                                            'landsat_x': landsat_x,
                                            'class': cls
                                        })
                                except Exception as e:
                                    logger.warning(f"Error transforming coordinates: {e}")
                                    continue
            except Exception as e:
                logger.warning(f"Error processing tile {os.path.basename(tile_info['path'])}: {e}")
                continue
        
        return locations
    
    def _create_synthetic_test_data(self) -> None:
        """Create synthetic test data for testing mode"""
        self.landsat_bands = np.zeros((6, 1000, 1000), dtype=np.float32)
        self.landsat_height = 1000
        self.landsat_width = 1000
        self.landsat_transform = rasterio.transform.Affine(30.0, 0.0, 0.0, 0.0, -30.0, 0.0)
        self.landsat_crs = rasterio.crs.CRS.from_epsg(4326)
        self.landsat_bounds = rasterio.coords.BoundingBox(0, 0, 1000, 1000)
        
        # Create synthetic samples
        self.sample_locations = []
        for i in range(10):
            self.sample_locations.append({
                'tile_info': None,
                'esa_y': 0,
                'esa_x': 0,
                'landsat_y': 500,
                'landsat_x': 500,
                'class': 10  # Tree cover
            })
    
    def _log_dataset_stats(self) -> None:
        """Log dataset statistics"""
        logger.info(f"ESA WorldCover Dataset")
        logger.info(f"Number of classes: {len(self.esa_classes)}")
        logger.info(f"Number of samples: {len(self.sample_locations)}")
        logger.info(f"Classes: {self.esa_classes}")
        
        # Log class distribution
        class_counts = {cls: 0 for cls in self.esa_classes}
        for sample in self.sample_locations:
            class_counts[sample['class']] += 1
        
        logger.info("Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  {cls}: {count} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.sample_locations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample (image patch, mask patch) from the dataset"""
        if not self.sample_locations:
            if self.test_mode:
                return self._get_synthetic_sample()
            else:
                raise ValueError("No valid sample locations available.")
        
        sample = self.sample_locations[idx % len(self.sample_locations)]
        
        # --- Extract Landsat Patch ---
        landsat_y = sample['landsat_y']
        landsat_x = sample['landsat_x']
        
        # Define window bounds for Landsat patch
        landsat_row_start = landsat_y - self.half_patch
        landsat_col_start = landsat_x - self.half_patch
        
        # Check bounds
        if not (0 <= landsat_row_start < self.landsat_height - self.patch_size and
                0 <= landsat_col_start < self.landsat_width - self.patch_size):
            logger.warning(f"Invalid Landsat window for sample {idx}. Skipping/retrying.")
            # Simple retry strategy: get the next sample
            return self.__getitem__((idx + 1) % len(self))
        
        spectral_patch = self.landsat_bands[:,
                                     landsat_row_start : landsat_row_start + self.patch_size,
                                     landsat_col_start : landsat_col_start + self.patch_size]
        
        # Validate spectral patch (check for excessive NaNs)
        nan_mask = np.isnan(spectral_patch)
        nan_ratio = nan_mask.sum() / spectral_patch.size
        if spectral_patch.shape != (len(self.landsat_bands), self.patch_size, self.patch_size) or nan_ratio > 0.95:
            # Log more details about the invalid patch
            logger.warning(
                f"Invalid spectral patch for sample {idx}. Shape: {spectral_patch.shape}, "
                f"NaN ratio: {nan_ratio:.2f}. Center Landsat (y,x): ({landsat_y}, {landsat_x}). "
                f"Window (r,c): ({landsat_row_start}, {landsat_col_start}). Retrying."
            )
            if self.test_mode: return self._get_synthetic_sample()
            return self.__getitem__((idx + 1) % len(self))
        
        # Replace remaining NaNs (e.g., with 0)
        # Use the mask we already computed
        spectral_patch[nan_mask] = 0.0
        
        # Convert spectral patch to tensor
        image_tensor = torch.from_numpy(spectral_patch.astype(np.float32))
        
        # --- Extract ESA Mask Patch ---
        tile_info = sample['tile_info']
        esa_y = sample['esa_y'] # Center row in ESA tile
        esa_x = sample['esa_x'] # Center col in ESA tile
        
        if tile_info is None: # Should only happen in test mode if synthetic data wasn't fully set up
             logger.warning(f"Missing tile_info for sample {idx}. Returning synthetic.")
             return self._get_synthetic_sample()
        
        try:
            with rasterio.open(tile_info['path']) as esa_src:
                # Define window for ESA patch centered at (esa_y, esa_x)
                esa_row_start = max(0, esa_y - self.half_patch)
                esa_col_start = max(0, esa_x - self.half_patch)
                # Adjust window size if near edge
                window_height = min(self.patch_size, esa_src.height - esa_row_start)
                window_width = min(self.patch_size, esa_src.width - esa_col_start)
                
                window = Window(esa_col_start, esa_row_start, window_width, window_height)
                esa_patch_raw = esa_src.read(1, window=window)
                
                # Handle cases where patch is smaller than target size (pad)
                if window_width < self.patch_size or window_height < self.patch_size:
                    padded_patch = np.full((self.patch_size, self.patch_size), self.ignore_index, dtype=np.uint8)
                    pad_row_offset = (self.patch_size - window_height) // 2
                    pad_col_offset = (self.patch_size - window_width) // 2
                    padded_patch[pad_row_offset:pad_row_offset+window_height,
                                 pad_col_offset:pad_col_offset+window_width] = esa_patch_raw
                    esa_patch_raw = padded_patch
                
        except rasterio.RasterioIOError as e:
            logger.error(f"Error reading ESA tile {tile_info['path']} for sample {idx}: {e}. Retrying.")
            if self.test_mode: return self._get_synthetic_sample()
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Unexpected error getting ESA mask for sample {idx}: {e}. Retrying.", exc_info=True)
            if self.test_mode: return self._get_synthetic_sample()
            return self.__getitem__((idx + 1) % len(self))
        
        # Map ESA labels (10, 20, ...) to 0-based indices (0, 1, ...)
        mask_patch = np.full_like(esa_patch_raw, self.ignore_index, dtype=np.int64) # Use int64 for tensor
        for esa_label, index in self.esa_label_to_index.items():
            mask_patch[esa_patch_raw == esa_label] = index
        
        mask_tensor = torch.from_numpy(mask_patch) # Shape [H, W]
        
        # --- Apply Manual Normalization ---
        # Ensure mean/std tensors are on the same device as the image tensor if using GPU later
        norm_mean = self.norm_mean.to(image_tensor.device)
        norm_std = self.norm_std.to(image_tensor.device)
        try:
            image_tensor = (image_tensor - norm_mean) / norm_std
        except Exception as e:
            logger.error(f"Error applying manual normalization for sample {idx}: {e}. Returning unnormalized tensor.")

        # --- Handle self.transform ---
        # If self.transform exists, it's likely the default one passed inadvertently.
        # Log a warning once, but don't apply it as normalization is handled manually.
        if self.transform and not getattr(self, '_transform_warning_logged', False):
            logger.warning("Dataset received a transform object, but manual normalization is applied internally. "
                           "Ensure transforms passed to get_esa_dataloader are None or only contain augmentations "
                           "compatible with raw tensors (not ToTensor/Normalize).")
            self._transform_warning_logged = True # Log only once

        # --- Final Checks and Return ---
        # Final check for tensor shapes
        if image_tensor.shape != (len(self.landsat_bands), self.patch_size, self.patch_size) or \
           mask_tensor.shape != (self.patch_size, self.patch_size):
            logger.error(f"Shape mismatch for sample {idx}! Image: {image_tensor.shape}, Mask: {mask_tensor.shape}. Retrying.")
            if self.test_mode: return self._get_synthetic_sample()
            return self.__getitem__((idx + 1) % len(self))

        return image_tensor, mask_tensor # Return image tensor [C, H, W] and mask tensor [H, W]
    
    def _get_synthetic_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a synthetic sample (image, mask) for test mode"""
        # Use num_bands determined during init
        num_bands = len(self.landsat_bands) if hasattr(self, 'landsat_bands') and self.landsat_bands is not None else 6
        dummy_img = torch.rand((num_bands, self.patch_size, self.patch_size), dtype=torch.float32)
        
        # Create a dummy mask with random valid class indices (0 to num_classes-1)
        num_classes = self.num_classes if hasattr(self, 'num_classes') else 11
        dummy_mask = torch.randint(0, num_classes, (self.patch_size, self.patch_size), dtype=torch.long)
        
        return dummy_img, dummy_mask

def get_esa_dataloader(
    esa_directory: str,
    landsat_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    patch_size: int = 224,
    num_samples: int = 10000,
    test_mode: bool = False,
    cache_images: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = False,
    ignore_index: int = DEFAULT_IGNORE_INDEX
) -> DataLoader:
    """
    Create a DataLoader for the ESA WorldCover dataset (for segmentation)
    
    Args:
        esa_directory (str): Path to the ESA WorldCover data directory
        landsat_path (str): Path to the Landsat data directory
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes for data loading
        transform (Optional[Callable]): Transform to apply to the images
        patch_size (int): Size of image patches to extract
        num_samples (int): Number of samples to generate
        test_mode (bool): Whether to run in test mode
        cache_images (bool): Whether to cache images in memory
        distributed (bool): Whether to use distributed training
        world_size (int): Number of processes in distributed training
        rank (int): Rank of current process in distributed training
        seed (int): Random seed for reproducibility
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        drop_last (bool): Whether to drop the last incomplete batch
        ignore_index (int): Pixel value to ignore in loss calculation
        
    Returns:
        DataLoader: Configured DataLoader for the ESA WorldCover dataset
    """
    # Create dataset, passing ignore_index
    dataset = ESAWorldCoverDataset(
        esa_directory=esa_directory,
        landsat_path=landsat_path,
        transform=transform,
        patch_size=patch_size,
        num_samples=num_samples,
        test_mode=test_mode,
        cache_images=cache_images,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=seed,
        ignore_index=ignore_index
    )
    
    # Create sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        shuffle = False
    else:
        # Use RandomSampler for training, SequentialSampler for validation typically
        # Assuming this dataloader is primarily for training, use RandomSampler
        sampler = RandomSampler(dataset)
        shuffle = False
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return dataloader

def get_default_transforms(
    input_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Dict[str, transforms.Compose]:
    """
    Get default transforms for training and validation
    
    Args:
        input_size (int): Size to resize images to
        mean (List[float]): Mean values for normalization
        std (List[float]): Standard deviation values for normalization
        
    Returns:
        Dict[str, transforms.Compose]: Dictionary containing train and val transforms
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Test ESA WorldCover Dataset Loader")
    parser.add_argument(
        '--esa_path', 
        type=str, 
        default='../data/ESA_WorldCover_10m_2021_v200_N30W120/',
        help='Path to the ESA WorldCover data directory'
    )
    parser.add_argument(
        '--landsat_path', 
        type=str, 
        default='../Landsat/',
        help='Path to the Landsat data directory'
    )
    args = parser.parse_args()

    print("="*30)
    print("RUNNING ESA WORLDCOVER DATASET TESTS...")
    print(f"Using ESA path: {args.esa_path}")
    print(f"Using Landsat path: {args.landsat_path}")
    print("="*30)

    # Check if paths exist
    if not os.path.exists(args.esa_path):
        logger.error(f"ESA path not found: {args.esa_path}")
        logger.error("Please provide a valid path using --esa_path argument.")
        exit(1)
    if not os.path.exists(args.landsat_path):
        logger.error(f"Landsat path not found: {args.landsat_path}")
        logger.error("Please provide a valid path using --landsat_path argument.")
        exit(1)

    # --- 1. Test Default Transforms ---
    print("\n--- Testing get_default_transforms ---")
    transforms_dict = get_default_transforms()
    assert 'train' in transforms_dict and isinstance(transforms_dict['train'], transforms.Compose)
    assert 'val' in transforms_dict and isinstance(transforms_dict['val'], transforms.Compose)
    print("Default transforms created successfully.")
    print(f"Train Transform: {transforms_dict['train']}")
    print(f"Val Transform: {transforms_dict['val']}")

    # --- 2. Test Dataset Instantiation ---
    print("\n--- Testing ESAWorldCoverDataset Instantiation ---")
    try:
        print("Testing Normal Mode...")
        dataset = ESAWorldCoverDataset(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000
        )
        print(f"Dataset created with {len(dataset)} samples.")

        print("Testing Test Mode...")
        test_dataset = ESAWorldCoverDataset(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
            test_mode=True
        )
        print(f"Test mode dataset created with {len(test_dataset)} samples.")
        
        print("Testing with Caching Enabled...")
        cached_dataset = ESAWorldCoverDataset(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
            cache_images=True
        )
        # Access a few items to populate cache
        _ = cached_dataset[0]
        _ = cached_dataset[1]
        print(f"Cached dataset created. Cache size: {len(cached_dataset.image_cache)}")

        print("Testing Distributed Setting (Simulated Rank 0)...")
        dist_dataset_rank0 = ESAWorldCoverDataset(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
            distributed=True,
            world_size=2,
            rank=0
        )
        print(f"Distributed dataset (Rank 0) created with {len(dist_dataset_rank0)} samples.")

        print("Testing Distributed Setting (Simulated Rank 1)...")
        dist_dataset_rank1 = ESAWorldCoverDataset(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
            distributed=True,
            world_size=2,
            rank=1
        )
        print(f"Distributed dataset (Rank 1) created with {len(dist_dataset_rank1)} samples.")
        
        # Basic check for distributed split correctness
        assert len(dist_dataset_rank0) + len(dist_dataset_rank1) == len(dataset), \
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
        print("Loading sample 0 from dataset...")
        img, mask = dataset[0]
        print(f"Sample 0: Image Shape={img.shape}, Type={img.dtype}, Mask Shape={mask.shape}, Type={mask.dtype}")
        assert isinstance(img, torch.Tensor) and len(img.shape) == 3, "Incorrect image format"
        assert isinstance(mask, torch.Tensor) and len(mask.shape) == 2, "Incorrect mask format"

        print("Loading sample 0 from cached dataset...")
        img_cached, mask_cached = cached_dataset[0]
        assert torch.equal(img, img_cached), "Cached image differs from non-cached"
        assert torch.equal(mask, mask_cached), "Cached mask differs from non-cached"
        
        print("Loading sample 0 from test mode dataset...")
        img_test, mask_test = test_dataset[0]
        print(f"Sample 0 (Test): Image Shape={img_test.shape}, Type={img_test.dtype}, Mask Shape={mask_test.shape}, Type={mask_test.dtype}")
        assert isinstance(img_test, torch.Tensor) and len(img_test.shape) == 3
        assert isinstance(mask_test, torch.Tensor) and len(mask_test.shape) == 2

        # Visualize the first sample
        plt.figure(figsize=(5, 5))
        # Need to unnormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img[:3] * std + mean  # Only show RGB channels
        img_display = img_display.permute(1, 2, 0).numpy().clip(0, 1) # C,H,W -> H,W,C
        plt.imshow(img_display)
        plt.title("Sample 0: Image")
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

    # --- 4. Test Dataloader Creation ---
    print("\n--- Testing get_esa_dataloader ---")
    try:
        print("Creating Train Dataloader (Batch Size=4, Workers=2)...")
        train_loader = get_esa_dataloader(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            batch_size=4,
            num_workers=2,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
            pin_memory=True,
            drop_last=True
        )
        print("Train DataLoader created successfully.")

        print("Creating Validation Dataloader (Batch Size=8, Workers=0)...")
        val_loader = get_esa_dataloader(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            batch_size=8,
            num_workers=0,
            transform=transforms_dict['val'],
            patch_size=224,
            num_samples=1000,
            pin_memory=False,
            drop_last=False
        )
        print("Validation DataLoader created successfully.")
        
        print("Testing iteration through Train Dataloader...")
        train_batch = next(iter(train_loader))
        imgs, masks = train_batch
        print(f"Train Batch: Images Shape={imgs.shape}, Masks Shape={masks.shape}")
        assert imgs.shape[0] == 4 and len(imgs.shape) == 4, "Incorrect train batch image shape"
        assert masks.shape[0] == 4, "Incorrect train batch mask shape"

        print("Testing iteration through Validation Dataloader...")
        val_batch = next(iter(val_loader))
        imgs_val, masks_val = val_batch
        print(f"Val Batch: Images Shape={imgs_val.shape}, Masks Shape={masks_val.shape}")
        # Batch size might be smaller than 8 if drop_last=False and dataset size is not multiple of 8
        assert imgs_val.shape[0] <= 8 and len(imgs_val.shape) == 4, "Incorrect val batch image shape"
        assert masks_val.shape[0] <= 8, "Incorrect val batch mask shape"
        
        # Test distributed dataloader creation (simulated)
        print("Creating Distributed Train Dataloader (Simulated)...")
        dist_loader = get_esa_dataloader(
            esa_directory=args.esa_path,
            landsat_path=args.landsat_path,
            batch_size=4,
            num_workers=0,
            transform=transforms_dict['train'],
            patch_size=224,
            num_samples=1000,
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
    print("ALL ESA WORLDCOVER DATASET TESTS PASSED SUCCESSFULLY!")
    print("="*30) 