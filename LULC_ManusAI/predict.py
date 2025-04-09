#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction system for Land Use/Land Cover Change and Wildfire Prediction
using the trained SAM2-UNet model
"""

import os
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from rasterio.plot import show
import logging
import time
from datetime import datetime
import glob

# Import our models
sys.path.append('/home/ubuntu/deep_learning_project')
from models.sam2_unet_lite import SAM2UNetLite, LandCoverChangeModelLite, WildfireDetectionModelLite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Prediction system for Land Use/Land Cover Change and Wildfire Prediction')
    parser.add_argument('--task', type=str, default='landcover', choices=['landcover', 'wildfire'],
                        help='Task to perform: landcover or wildfire')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='Size of image patches for processing')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between adjacent patches')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for prediction')
    parser.add_argument('--time-series', action='store_true',
                        help='Process input as time series for change detection')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of predictions')
    return parser.parse_args()

def load_model(args):
    """Load model from checkpoint"""
    logger.info(f"Loading model for {args.task} task from {args.checkpoint}")
    
    try:
        if args.task == 'landcover':
            model = LandCoverChangeModelLite(num_classes=21)  # UC Merced has 21 land use classes
        elif args.task == 'wildfire':
            model = WildfireDetectionModelLite(num_classes=2)  # Binary classification: fire/no-fire
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from epoch {checkpoint['epoch']+1}")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_image(image_path, task='landcover'):
    """Load and preprocess an image for prediction"""
    logger.info(f"Loading image: {image_path}")
    
    try:
        # Check if the file is a GeoTIFF
        if image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                # Read image data
                if task == 'landcover':
                    # For landcover, use RGB bands
                    if src.count >= 3:
                        image = src.read([1, 2, 3])  # RGB bands
                        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
                    else:
                        # If less than 3 bands, duplicate the first band
                        image = src.read(1)
                        image = np.stack([image, image, image], axis=2)
                elif task == 'wildfire':
                    # For wildfire, use RGB + NBR if available
                    if src.count >= 4:
                        image = src.read([1, 2, 3, 4])  # RGB + NBR bands
                        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
                    else:
                        # If less than 4 bands, use available bands and pad
                        bands = [src.read(i+1) for i in range(src.count)]
                        while len(bands) < 4:
                            bands.append(np.zeros_like(bands[0]))
                        image = np.stack(bands, axis=2)
                
                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height
                }
        else:
            # Regular image file
            image = np.array(Image.open(image_path).convert('RGB'))
            
            # For wildfire task, create a placeholder NBR band
            if task == 'wildfire':
                nbr = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                image = np.dstack([image, nbr])
            
            metadata = {
                'transform': None,
                'crs': None,
                'width': image.shape[1],
                'height': image.shape[0]
            }
        
        return image, metadata
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None, None

def preprocess_image(image, patch_size=128, overlap=32, task='landcover'):
    """Preprocess image into patches for prediction"""
    logger.info(f"Preprocessing image of shape {image.shape}")
    
    try:
        height, width = image.shape[:2]
        
        # Calculate stride (patch_size - overlap)
        stride = patch_size - overlap
        
        # Calculate number of patches in each dimension
        n_h = 1 + (height - patch_size) // stride if height > patch_size else 1
        n_w = 1 + (width - patch_size) // stride if width > patch_size else 1
        
        # Adjust last patch to include the image boundary
        if height > patch_size and height - (stride * (n_h - 1) + patch_size) > 0:
            n_h += 1
        if width > patch_size and width - (stride * (n_w - 1) + patch_size) > 0:
            n_w += 1
        
        patches = []
        patch_locations = []
        
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates
                h_start = min(i * stride, height - patch_size)
                w_start = min(j * stride, width - patch_size)
                h_end = h_start + patch_size
                w_end = w_start + patch_size
                
                # Handle boundary cases
                if h_end > height:
                    h_start = max(0, height - patch_size)
                    h_end = height
                if w_end > width:
                    w_start = max(0, width - patch_size)
                    w_end = width
                
                # Extract patch
                patch = image[h_start:h_end, w_start:w_end]
                
                # Pad if necessary
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    pad_h = max(0, patch_size - patch.shape[0])
                    pad_w = max(0, patch_size - patch.shape[1])
                    
                    if len(patch.shape) == 3:  # RGB or RGB+NBR
                        padded_patch = np.pad(
                            patch, 
                            ((0, pad_h), (0, pad_w), (0, 0)), 
                            mode='constant'
                        )
                    else:  # Grayscale
                        padded_patch = np.pad(
                            patch, 
                            ((0, pad_h), (0, pad_w)), 
                            mode='constant'
                        )
                    
                    patch = padded_patch
                
                # Normalize patch
                if task == 'landcover':
                    # Normalize RGB
                    patch = patch.astype(np.float32) / 255.0
                    if len(patch.shape) == 3 and patch.shape[2] == 3:
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        patch = (patch - mean) / std
                elif task == 'wildfire':
                    # Normalize RGB+NBR
                    if len(patch.shape) == 3 and patch.shape[2] >= 3:
                        # Normalize RGB part
                        patch[:, :, :3] = patch[:, :, :3].astype(np.float32) / 255.0
                        # NBR is already normalized or synthetic
                    
                    # Apply standard normalization
                    patch = (patch - 0.5) / 0.5
                
                # Convert to tensor and add batch dimension
                if len(patch.shape) == 3:
                    patch = torch.from_numpy(patch).permute(2, 0, 1).float()  # (C, H, W)
                else:
                    patch = torch.from_numpy(patch).unsqueeze(0).float()  # (1, H, W)
                
                patches.append(patch)
                patch_locations.append((h_start, w_start, h_end, w_end))
        
        return patches, patch_locations, (height, width)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None, None

def predict_patches(model, patches, device, batch_size=4, task='landcover'):
    """Run prediction on image patches"""
    logger.info(f"Predicting {len(patches)} patches with batch size {batch_size}")
    
    try:
        predictions = []
        
        # Process patches in batches
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            batch_tensor = torch.stack(batch_patches).to(device)
            
            # Create a duplicate batch for time series input (placeholder)
            batch_tensor2 = batch_tensor.clone()
            
            with torch.no_grad():
                if task == 'landcover':
                    # Forward pass
                    seg1, seg2, change_prob = model(batch_tensor, batch_tensor2)
                    
                    # Get class predictions
                    _, preds = torch.max(seg1, 1)
                    
                elif task == 'wildfire':
                    # Forward pass
                    fire1, fire2, spread_prob = model(batch_tensor, batch_tensor2)
                    
                    # Get class predictions
                    _, preds = torch.max(fire1, 1)
            
            # Convert predictions to numpy
            batch_preds = preds.cpu().numpy()
            predictions.extend([pred for pred in batch_preds])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error predicting patches: {e}")
        return None

def stitch_predictions(predictions, patch_locations, image_size, overlap=32):
    """Stitch patch predictions back into a full image"""
    logger.info(f"Stitching predictions into image of size {image_size}")
    
    try:
        height, width = image_size
        full_pred = np.zeros((height, width), dtype=np.uint8)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Create a weight map for blending overlapping regions
        # Higher weight in the center of the patch, lower at the edges
        patch_size = predictions[0].shape[0]
        y, x = np.mgrid[0:patch_size, 0:patch_size]
        center_y, center_x = patch_size // 2, patch_size // 2
        # Gaussian weight
        weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (patch_size // 4)**2))
        
        # Stitch patches
        for pred, (h_start, w_start, h_end, w_end) in zip(predictions, patch_locations):
            # Get actual patch dimensions
            h_size = h_end - h_start
            w_size = w_end - w_start
            
            # Handle padded patches
            if pred.shape[0] > h_size or pred.shape[1] > w_size:
                pred = pred[:h_size, :w_size]
                weight_patch = weight[:h_size, :w_size]
            else:
                weight_patch = weight
            
            # Apply weighted blending
            full_pred[h_start:h_end, w_start:w_end] += pred * weight_patch
            weight_map[h_start:h_end, w_start:w_end] += weight_patch
        
        # Normalize by weight map
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        full_pred = (full_pred / weight_map).astype(np.uint8)
        
        return full_pred
    
    except Exception as e:
        logger.error(f"Error stitching predictions: {e}")
        return None

def save_prediction(prediction, output_path, metadata=None, colormap=None, task='landcover'):
    """Save prediction as GeoTIFF or image file"""
    logger.info(f"Saving prediction to {output_path}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as GeoTIFF if metadata is available
        if metadata and metadata['transform'] is not None and metadata['crs'] is not None:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=prediction.dtype,
                crs=metadata['crs'],
                transform=metadata['transform']
            ) as dst:
                dst.write(prediction, 1)
        else:
            # Save as regular image
            if task == 'landcover':
                # Use a colormap for land cover classes
                cmap = plt.cm.get_cmap('tab20', 21)
                colored_pred = cmap(prediction)
                colored_pred = (colored_pred[:, :, :3] * 255).astype(np.uint8)
                Image.fromarray(colored_pred).save(output_path)
            elif task == 'wildfire':
                # Use a fire-like colormap for wildfire
                cmap = plt.cm.get_cmap('hot')
                colored_pred = cmap(prediction)
                colored_pred = (colored_pred[:, :, :3] * 255).astype(np.uint8)
                Image.fromarray(colored_pred).save(output_path)
        
        logger.info(f"Prediction saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return False

def visualize_prediction(image, prediction, output_path, task='landcover'):
    """Create visualization of input image and prediction"""
    logger.info(f"Creating visualization for {output_path}")
    
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot input image
        plt.subplot(1, 2, 1)
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Show RGB channels
            rgb = image[:, :, :3]
            # Denormalize if needed
            if rgb.max() <= 1.0:
                rgb = rgb * 255
            plt.imshow(rgb.astype(np.uint8))
        else:
            plt.imshow(image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        # Plot prediction
        plt.subplot(1, 2, 2)
        if task == 'landcover':
            plt.imshow(prediction, cmap='tab20')
            plt.title('Land Cover Prediction')
        elif task == 'wildfire':
            plt.imshow(prediction, cmap='hot')
            plt.title('Wildfire Prediction')
        plt.axis('off')
     
(Content truncated due to size limit. Use line ranges to read in chunks)