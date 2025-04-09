import torch
import numpy as np
import rasterio
from rasterio.warp import transform
import geopandas as gpd
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any
import os
import glob
from tqdm import tqdm

logger = logging.getLogger(__name__)

def preprocess_landsat_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocess a Landsat patch for model input.
    
    Args:
        patch: Raw Landsat patch (C, H, W)
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Convert to float32
    patch = patch.astype(np.float32)
    
    # Scale to [0, 1] range
    # Landsat surface reflectance values are typically in [0, 10000]
    patch = patch / 10000.0
    
    # Handle nodata values
    patch[patch < 0] = 0
    patch[patch > 1] = 1
    
    # Convert to tensor and add batch dimension
    patch_tensor = torch.from_numpy(patch).float()
    return patch_tensor.unsqueeze(0)

def postprocess_output(output_tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess model output to get class predictions.
    
    Args:
        output_tensor: Model output tensor (B, C, H, W)
        
    Returns:
        Class prediction array (H, W)
    """
    # Get class predictions
    pred = torch.argmax(output_tensor, dim=1)
    return pred.squeeze().cpu().numpy()

def run_inference_on_raster(
    model: torch.nn.Module,
    device: torch.device,
    raster_path: str,
    output_map_path: str,
    batch_size: int = 16,
    patch_size: int = 256,
    stride: int = 128
) -> str:
    """
    Run inference on a large raster file using a sliding window approach.
    
    Args:
        model: Trained PyTorch model
        device: Device to run inference on
        raster_path: Path to input raster file
        output_map_path: Path to save output segmentation map
        batch_size: Number of patches to process in parallel
        patch_size: Size of square patches for inference
        stride: Stride for sliding window
        
    Returns:
        Path to saved segmentation map
    """
    model.eval()
    
    with rasterio.open(raster_path) as src:
        # Get raster metadata
        profile = src.profile
        height, width = src.shape
        num_bands = src.count
        
        # Update profile for output
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )
        
        # Initialize output array
        output_map = np.zeros((height, width), dtype=np.uint8)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        
        # Process patches
        patches = []
        coords = []
        
        for y in tqdm(range(0, height - patch_size + 1, stride), desc="Processing rows"):
            for x in range(0, width - patch_size + 1, stride):
                # Read patch
                window = rasterio.windows.Window(x, y, patch_size, patch_size)
                patch = src.read(window=window)
                
                # Skip if too many nodata values
                if np.isnan(patch).sum() > 0.1 * patch.size:
                    continue
                
                # Preprocess patch
                patch_tensor = preprocess_landsat_patch(patch)
                patches.append(patch_tensor)
                coords.append((y, x))
                
                # Process batch when full
                if len(patches) == batch_size:
                    with torch.no_grad():
                        batch = torch.cat(patches, dim=0).to(device)
                        outputs = model(batch)
                        preds = postprocess_output(outputs)
                        
                        # Update output map
                        for i, (y_coord, x_coord) in enumerate(coords):
                            output_map[y_coord:y_coord+patch_size, x_coord:x_coord+patch_size] = preds[i]
                    
                    patches = []
                    coords = []
        
        # Process remaining patches
        if patches:
            with torch.no_grad():
                batch = torch.cat(patches, dim=0).to(device)
                outputs = model(batch)
                preds = postprocess_output(outputs)
                
                for i, (y_coord, x_coord) in enumerate(coords):
                    output_map[y_coord:y_coord+patch_size, x_coord:x_coord+patch_size] = preds[i]
        
        # Save output map
        os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
        with rasterio.open(output_map_path, 'w', **profile) as dst:
            dst.write(output_map, 1)
    
    return output_map_path

def apply_model_to_landsat(
    model: torch.nn.Module,
    landsat_path: str,
    shapefile_path: str,
    output_path: str,
    test_mode: bool = False
) -> None:
    """
    Apply a trained model to Landsat imagery.
    
    Args:
        model: Trained model
        landsat_path: Path to Landsat data directory
        shapefile_path: Path to shapefile defining area of interest
        output_path: Path to save output predictions
        test_mode: Whether to run in test mode
    """
    # Load shapefile if provided
    if shapefile_path and os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        bounds = gdf.total_bounds
    else:
        bounds = None
    
    # Load Landsat bands
    landsat_files = sorted(glob.glob(os.path.join(landsat_path, '*_SR_B*.TIF')))
    if not landsat_files:
        raise ValueError(f"No Landsat SR bands found in {landsat_path}")
    
    # For LULC, we want RGB + NIR + SWIR bands
    band_numbers = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']  # B, G, R, NIR, SWIR1, SWIR2
    selected_bands = []
    
    for band_num in band_numbers:
        matching_files = [f for f in landsat_files if f'_SR_{band_num}.' in f]
        if matching_files:
            selected_bands.append(matching_files[0])
        else:
            logger.warning(f"Could not find {band_num} band")
    
    # Load bands and get metadata
    bands_data = []
    for band_file in selected_bands:
        with rasterio.open(band_file) as src:
            band_data = src.read(1).astype(np.float32)
            band_data = band_data / 10000.0  # Scale to reflectance
            
            if src.nodata is not None:
                band_data[band_data == src.nodata] = np.nan
            
            bands_data.append(band_data)
            
            # Store metadata from first band
            if len(bands_data) == 1:
                transform = src.transform
                crs = src.crs
                width = src.width
                height = src.height
    
    # Stack bands
    landsat_data = np.stack(bands_data)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process in patches
    patch_size = 256
    stride = 128
    
    # Initialize output array
    output = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    
    # Process patches
    for y in tqdm(range(0, height - patch_size + 1, stride), desc="Processing rows"):
        for x in range(0, width - patch_size + 1, stride):
            # Extract patch
            patch = landsat_data[:, y:y+patch_size, x:x+patch_size]
            
            # Skip if too many NaN values
            if np.isnan(patch).sum() > 0.1 * patch.size:
                continue
            
            # Normalize and convert to tensor
            patch = np.nan_to_num(patch)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
            
            # Predict
            with torch.no_grad():
                model.eval()
                output_tensor = model(patch_tensor)
                pred = output_tensor.argmax(dim=1).item()
                conf = torch.softmax(output_tensor, dim=1).max().item()
            
            # Update output arrays
            output[y:y+patch_size, x:x+patch_size] = pred
            confidence[y:y+patch_size, x:x+patch_size] = conf
    
    # Save output
    output_file = os.path.join(output_path, 'predictions.tif')
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(output, 1)
    
    # Save confidence
    confidence_file = os.path.join(output_path, 'confidence.tif')
    with rasterio.open(
        confidence_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(confidence, 1)
    
    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Confidence map saved to {confidence_file}") 