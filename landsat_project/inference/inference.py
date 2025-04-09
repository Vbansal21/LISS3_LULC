import os
import torch
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
import json
import math
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F # For softmax

from models.model_factory import ModelFactory
from models.model_config import ModelConfig
from config import MODEL, VISUALIZATION, LOGGING, DATASET, TESTING # Import TESTING for window size/stride
from utils.logging import Logger # Assuming Logger is correctly defined

# Use the configured logger if available, otherwise basic config
try:
    logger = Logger(LOGGING)
except NameError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# --- Helper Functions --- 

def parse_mtl_file(mtl_path: str) -> Dict:
    """Parse Landsat MTL metadata file (text key-value format)."""
    metadata = {}
    try:
        with open(mtl_path, 'r') as f:
            current_group = None
            group_data = {}
            for line in f:
                line = line.strip()
                if not line or line == 'END':
                    continue
                if line.startswith('GROUP = '):
                    current_group = line.split('=')[1].strip()
                    group_data = {}
                    metadata[current_group] = group_data
                elif line.startswith('END_GROUP = '):
                    current_group = None
                elif '=' in line:
                    parts = line.split('=', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Remove quotes
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Try converting to number
                    try:
                        if '.' in value:
                            parsed_value = float(value)
                        else:
                            parsed_value = int(value)
                    except ValueError:
                        parsed_value = value
                        
                    if current_group:
                         group_data[key] = parsed_value
                    else: # Store top-level items directly if any
                         metadata[key] = parsed_value
                         
    except FileNotFoundError:
        logger.error(f"MTL file not found at: {mtl_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing MTL file {mtl_path}: {e}")
        # Return empty dict or raise, depending on desired behavior
        pass # Or raise e
    
    # Flatten specific nested groups if needed, e.g., IMAGE_ATTRIBUTES
    flat_metadata = {}
    for group, data in metadata.items():
        if isinstance(data, dict):
            for key, value in data.items():
                flat_metadata[f"{group}.{key}"] = value
        else:
             flat_metadata[group] = data # Keep top-level items
             
    return flat_metadata

def load_landsat_bands(scene_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Finds the first band file (to get profile) and parses metadata.
       Returns path to first band file and metadata dict.
    """
    scene_path = Path(scene_dir)
    logger.info(f"Searching for Landsat bands and MTL file in: {scene_path}")

    # Find MTL file
    mtl_files = list(scene_path.glob('*_MTL.txt'))
    if not mtl_files:
        logger.error(f"MTL file (*_MTL.txt) not found in {scene_dir}")
        raise FileNotFoundError(f"MTL file not found in {scene_dir}")
    if len(mtl_files) > 1:
        logger.warning(f"Multiple MTL files found in {scene_dir}, using the first one: {mtl_files[0]}")
    mtl_path = mtl_files[0]
    logger.info(f"Found MTL file: {mtl_path}")

    # Parse metadata
    metadata = {}
    try:
        metadata = parse_mtl_file(str(mtl_path))
        if not metadata:
             logger.warning(f"MTL file parsing resulted in empty metadata for {mtl_path}")
        else:
             logger.info(f"Successfully parsed metadata from {mtl_path}")
    except Exception as e:
        logger.error(f"Failed to parse MTL file {mtl_path}: {e}")
        metadata = {}

    # Find the first band file (e.g., B1 or B2) to get profile later
    first_band_file = None
    for band_num in range(1, 8):
        band_suffix = f"SR_B{band_num}"
        glob_pattern = f"*{band_suffix}.TIF"
        band_files = list(scene_path.glob(glob_pattern))
        if not band_files:
            alt_pattern_l8 = f"LC08_*_{band_suffix}.TIF"
            alt_pattern_l9 = f"LC09_*_{band_suffix}.TIF"
            band_files = list(scene_path.glob(alt_pattern_l8)) + list(scene_path.glob(alt_pattern_l9))
        
        if band_files:
            first_band_file = str(band_files[0])
            logger.info(f"Found first band file for profile: {first_band_file}")
            break # Stop once the first band is found
            
    if not first_band_file:
        logger.error(f"Could not find any Landsat SR band files (B1-B7) in {scene_dir}")
        raise FileNotFoundError(f"No Landsat SR band files found in {scene_dir}")

    return first_band_file, metadata

def preprocess_patch(patch_np: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> torch.Tensor:
    """Preprocess a single patch (window) of Landsat data."""
    if patch_np is None or patch_np.size == 0:
        raise ValueError("Input patch to preprocess_patch is invalid")

    # Convert to float32
    patch_np = patch_np.astype(np.float32)

    # --- Normalization --- 
    if mean is not None and std is not None:
        if len(mean) == patch_np.shape[0] and len(std) == patch_np.shape[0]:
            mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
            std[std < 1e-6] = 1e-6 # Avoid division by zero
            patch_np = (patch_np - mean) / std
        else:
            logger.warning(f"Mean/Std length mismatch ({len(mean)}/{len(std)}) vs patch channels ({patch_np.shape[0]}). Using fallback min-max scaling.")
            mean, std = None, None # Force fallback
    
    if mean is None or std is None:
        # Fallback: Min-Max scaling per band 
        # logger.debug("Performing Min-Max scaling per band for patch.")
        for band_idx in range(patch_np.shape[0]):
            band_data = patch_np[band_idx]
            # Handle potential all-NaN slices if necessary, though NaNs should ideally be handled by rasterio's read
            valid_mask = np.isfinite(band_data)
            if np.any(valid_mask):
                min_val = np.min(band_data[valid_mask])
                max_val = np.max(band_data[valid_mask])
                if max_val > min_val:
                    patch_np[band_idx] = (band_data - min_val) / (max_val - min_val)
                elif max_val == min_val:
                    patch_np[band_idx] = 0.0 # Constant value band -> set to 0
            else:
                 patch_np[band_idx] = 0.0 # All NaN band -> set to 0

    # Convert to tensor and add batch dimension
    patch_tensor = torch.from_numpy(patch_np).unsqueeze(0)
    return patch_tensor

def generate_colored_prediction(predictions: np.ndarray, class_colors_map: Dict) -> Image.Image:
    """Generates a colored image from a prediction map using a color dictionary."""
    height, width = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Use ESA colors by default, handle missing keys gracefully
    default_color = (0, 0, 0) # Black for unknown classes
    esa_colors = class_colors_map.get('esa', {})
    
    unique_classes = np.unique(predictions)
    logger.debug(f"Unique prediction classes found: {unique_classes}")
    
    for class_id in unique_classes:
        color = esa_colors.get(class_id, default_color)
        mask = predictions == class_id
        rgb_image[mask] = color
        # logger.debug(f"Class {class_id}: Color {color}, Pixels {np.sum(mask)}")

    return Image.fromarray(rgb_image, 'RGB')

def run_segmentation_inference(
    model: torch.nn.Module,
    landsat_scene_path: str,
    output_dir: Path,
    config: ModelConfig,
    device: torch.device,
    test_mode: bool = False, # Not directly used here, but kept for consistency
    batch_size: int = 1 # Batch size for sliding window (process multiple patches at once)
) -> None:
    """Run sliding window inference on a Landsat scene, save results & visualizations."""
    logger.info(f"Starting sliding window inference for Landsat scene: {landsat_scene_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration --- 
    patch_size = TESTING['sliding_window'].get('size', DATASET.get('patch_size', 256)) # Use testing or dataset config
    stride = TESTING['sliding_window'].get('stride', patch_size // 2) # Default stride is half the patch size
    in_channels = config.in_channels
    num_classes = config.num_classes
    # Ensure mean/std match input channels
    norm_mean = DATASET['normalization']['mean'][:in_channels]
    norm_std = DATASET['normalization']['std'][:in_channels]
    if len(norm_mean) != in_channels or len(norm_std) != in_channels:
        logger.warning(f"Config mean/std length mismatch ({len(norm_mean)}/{len(norm_std)}) vs model input channels ({in_channels}). Check config.py. Using fallback normalization.")
        norm_mean, norm_std = None, None # Fallback to min-max scaling in preprocess_patch
    else:
         logger.info(f"Using mean/std normalization with {in_channels} channels.")

    logger.info(f"Using Patch Size: {patch_size}, Stride: {stride}, Batch Size: {batch_size}")

    try:
        # --- Load Data Info --- 
        first_band_path, metadata = load_landsat_bands(str(landsat_scene_path))
        scene_name = Path(landsat_scene_path).name

        # Open the first band file to get dimensions and profile
        with rasterio.open(first_band_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            logger.info(f"Image dimensions: {width}x{height}")
            if profile['count'] != 1:
                 logger.warning(f"Profile count is {profile['count']}, expected 1 for single band. Adjusting profile.")
                 profile['count'] = 1 # Ensure profile is for single band output initially

        # --- Initialize Output Arrays --- 
        # Use float32 for accumulation to allow averaging
        full_predictions = np.zeros((height, width), dtype=np.float32)
        full_confidence = np.zeros((height, width), dtype=np.float32)
        window_counts = np.zeros((height, width), dtype=np.uint8)

        # --- Sliding Window --- 
        model.eval()
        windows = []
        for r in range(0, height - patch_size + 1, stride):
            for c in range(0, width - patch_size + 1, stride):
                windows.append(Window(c, r, patch_size, patch_size))
        # Add edge windows if stride doesn't perfectly align
        if height % stride != 0:
             r = height - patch_size
             for c in range(0, width - patch_size + 1, stride): windows.append(Window(c, r, patch_size, patch_size))
        if width % stride != 0:
             c = width - patch_size
             for r in range(0, height - patch_size + 1, stride): windows.append(Window(c, r, patch_size, patch_size))
        if height % stride != 0 and width % stride != 0:
             windows.append(Window(width - patch_size, height - patch_size, patch_size, patch_size))
        
        # Remove duplicate windows (if any caused by edge cases)
        windows = list(set(windows))
        logger.info(f"Total windows to process: {len(windows)}")

        # Determine required bands (e.g., B2-B7 for 6 channels)
        # Assuming bands B2, B3, B4, B5, B6, B7 correspond to indices 1-6 if files are B1-B7
        required_band_indices = list(range(1, in_channels + 1)) # e.g., [1, 2, 3, 4, 5, 6] for 6 channels
        logger.info(f"Model requires {in_channels} channels. Reading rasterio bands: {required_band_indices}")

        batch_patches = []
        batch_windows = []
        
        # --- Process Windows in Batches --- 
        with rasterio.open(first_band_path) as src: # Re-open to read with windows
             # We need to read *all* required bands for each window
             # This is inefficient if done window-by-window. Ideally, read larger chunks if memory allows.
             # For simplicity now, we read per-window, assuming files are band-interleaved or accessible.
             # This assumes the `src` opened above (first band) has the same spatial properties as other bands.
             # A more robust approach opens each band file separately or uses a VRT.
             # Let's assume for now we can read required bands from the *scene directory* for each window.

             for i, window in enumerate(tqdm(windows, desc="Processing Windows")):
                 try:
                     # Read *all* required bands for this window
                     window_bands_data = []
                     for band_num_idx in required_band_indices:
                         band_suffix = f"SR_B{band_num_idx+1}" # +1 because indices are 0-based, bands 1-based
                         glob_pattern = f"*{band_suffix}.TIF"
                         band_files = list(Path(landsat_scene_path).glob(glob_pattern))
                         if not band_files:
                             alt_pattern_l8 = f"LC08_*_{band_suffix}.TIF"
                             alt_pattern_l9 = f"LC09_*_{band_suffix}.TIF"
                             band_files = list(Path(landsat_scene_path).glob(alt_pattern_l8)) + list(Path(landsat_scene_path).glob(alt_pattern_l9))
                         
                         if not band_files:
                             raise FileNotFoundError(f"Required band file {band_suffix} not found for window {window}")
                             
                         with rasterio.open(band_files[0]) as band_src:
                              # Read the specific window from this band file
                              band_data = band_src.read(1, window=window)
                              window_bands_data.append(band_data)
                              
                     # Stack the bands for this window
                     patch_np = np.stack(window_bands_data, axis=0)
                     
                     # Preprocess the patch
                     patch_tensor = preprocess_patch(patch_np, norm_mean, norm_std)
                     batch_patches.append(patch_tensor)
                     batch_windows.append(window)

                 except FileNotFoundError as e:
                      logger.warning(f"Skipping window {window} due to missing band file: {e}")
                      continue
                 except Exception as e:
                      logger.warning(f"Skipping window {window} due to error reading/stacking bands: {e}")
                      continue

                 # Process batch when full or at the end
                 if len(batch_patches) == batch_size or i == len(windows) - 1:
                     if not batch_patches: continue # Skip if batch is empty
                     
                     batch_tensors = torch.cat(batch_patches, dim=0).to(device)
                     
                     with torch.no_grad():
                         output_logits = model(batch_tensors) # [B, C, H, W]
                         if isinstance(output_logits, tuple): output_logits = output_logits[0]
                         
                         probabilities = F.softmax(output_logits, dim=1) # [B, C, H, W]
                         batch_confidence, batch_predictions = torch.max(probabilities, dim=1) # [B, H, W]
                         
                         batch_predictions_np = batch_predictions.cpu().numpy()
                         batch_confidence_np = batch_confidence.cpu().numpy()

                     # Add batch results to full arrays
                     for j in range(len(batch_patches)):
                         win = batch_windows[j]
                         pred = batch_predictions_np[j]
                         conf = batch_confidence_np[j]
                         
                         # Add results, handling overlaps by summation for now
                         full_predictions[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width] += pred
                         full_confidence[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width] += conf
                         window_counts[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width] += 1

                     # Clear batch
                     batch_patches = []
                     batch_windows = []
        
        # --- Finalize Output --- 
        # Average overlapping areas
        valid_mask = window_counts > 0
        full_predictions[valid_mask] = full_predictions[valid_mask] / window_counts[valid_mask]
        full_confidence[valid_mask] = full_confidence[valid_mask] / window_counts[valid_mask]
        # Round predictions to nearest integer class after averaging
        full_predictions = np.round(full_predictions).astype(np.uint8)
        # Fill areas with no predictions (count == 0) with a nodata value if desired
        # full_predictions[~valid_mask] = 255 # Example: Use 255 as nodata
        # full_confidence[~valid_mask] = 0.0

        # --- Save Output GeoTIFFs --- 
        pred_filename = output_dir / f"{scene_name}_prediction.tif"
        conf_filename = output_dir / f"{scene_name}_confidence.tif"
        
        # Update profile for prediction map (uint8)
        profile.update(dtype=rasterio.uint8, count=1, nodata=None) # Set nodata if needed
        logger.info(f"Saving prediction map (uint8) to: {pred_filename}")
        with rasterio.open(pred_filename, 'w', **profile) as dst:
            dst.write(full_predictions, 1)
            
        # Update profile for confidence map (float32)
        profile.update(dtype=rasterio.float32, count=1, nodata=None)
        logger.info(f"Saving confidence map (float32) to: {conf_filename}")
        with rasterio.open(conf_filename, 'w', **profile) as dst:
            dst.write(full_confidence, 1)
            
        # --- Save Visualizations (JPEG) --- 
        pred_jpeg_filename = output_dir / f"{scene_name}_prediction_color.jpg"
        conf_jpeg_filename = output_dir / f"{scene_name}_confidence_gray.jpg"
        
        try:
            # Generate colored prediction image
            color_pred_img = generate_colored_prediction(full_predictions, VISUALIZATION['class_colors'])
            logger.info(f"Saving colored prediction JPEG to: {pred_jpeg_filename}")
            color_pred_img.save(pred_jpeg_filename, quality=90)
        except Exception as e:
             logger.error(f"Failed to save colored prediction JPEG: {e}")
             
        try:
             # Generate grayscale confidence image
             # Scale confidence 0-1 to 0-255
             conf_scaled = (full_confidence * 255).astype(np.uint8)
             conf_img = Image.fromarray(conf_scaled, 'L') # 'L' mode for grayscale
             logger.info(f"Saving grayscale confidence JPEG to: {conf_jpeg_filename}")
             conf_img.save(conf_jpeg_filename, quality=90)
        except Exception as e:
             logger.error(f"Failed to save confidence JPEG: {e}")

        # --- Save Metadata --- 
        if metadata:
            metadata_filename = output_dir / f"{scene_name}_metadata.json"
            try:
                serializable_metadata = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in metadata.items()}
                with open(metadata_filename, 'w') as f:
                    json.dump(serializable_metadata, f, indent=4)
                logger.info(f"Saving metadata to: {metadata_filename}")
            except Exception as e:
                 logger.error(f"Failed to save metadata JSON: {e}")
                 
        logger.info(f"Sliding window inference completed for {scene_name}.")

    # --- Error Handling --- 
    except FileNotFoundError as e:
        logger.error(f"File not found during inference for {landsat_scene_path}: {e}")
        raise
    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio error during inference for {landsat_scene_path}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error during inference (e.g., shape mismatch, invalid patch) for {landsat_scene_path}: {e}")
        raise
    except RuntimeError as e:
         logger.error(f"Runtime error during inference (e.g., CUDA OOM) for {landsat_scene_path}: {e}")
         raise
    except Exception as e:
        logger.error(f"Unexpected error during inference for {landsat_scene_path}: {e}")
        raise

# --- Standalone Execution Logic --- 

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint and prepare for inference."""
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    try:
        # Handle safe loading if ModelConfig is stored in checkpoint
        safe_globals = {} # Define empty or current globals
        try: safe_globals = torch.serialization.get_safe_globals()
        except AttributeError: pass # Ignore if function doesn't exist
        if ModelConfig not in safe_globals.get('_safe_loaded_classes', []):
            try: torch.serialization.add_safe_globals([ModelConfig])
            except AttributeError: logger.warning("Could not add ModelConfig to safe globals.")
            except Exception as e: logger.warning(f"Error adding ModelConfig to safe globals: {e}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load config from checkpoint if available
        if 'config' in checkpoint and isinstance(checkpoint['config'], ModelConfig):
             model_config = checkpoint['config']
             logger.info("Loaded ModelConfig from checkpoint.")
        elif 'config' in checkpoint and isinstance(checkpoint['config'], dict):
             logger.warning("Loading ModelConfig from dict in checkpoint. May lack defaults.")
             try:
                 # Attempt to create config, assuming necessary keys are present
                 model_config = ModelConfig(**checkpoint['config'])
             except TypeError as te:
                  logger.error(f"Failed to create ModelConfig from checkpoint dict: {te}. Checkpoint config: {checkpoint['config']}")
                  raise ValueError("Could not load model config from checkpoint dict.") from te
        else:
             # Attempt to infer config based on state_dict keys (less reliable)
             logger.warning("Model config not found in checkpoint. Attempting to create a default config.")
             # This requires knowing num_classes and in_channels, difficult without config!
             # We might need to pass these as arguments or read from state_dict structure.
             # For now, raise an error or use hardcoded defaults.
             # Example: Get num_classes from the final layer size in state_dict
             # state_dict = checkpoint.get('model_state_dict', checkpoint)
             # final_layer_key = list(state_dict.keys())[-1] # Risky assumption
             # num_classes_inferred = state_dict[final_layer_key].shape[0]
             # in_channels_inferred = state_dict[list(state_dict.keys())[0]].shape[1] # Risky
             # model_config = ModelConfig(num_classes=num_classes_inferred, in_channels=in_channels_inferred) # Need other defaults!
             raise ValueError("Cannot load model: Config missing from checkpoint and cannot be reliably inferred.")
             
        # Load model state
        model = ModelFactory.create_model(model_config)
        if 'model_state_dict' in checkpoint:
             state_dict = checkpoint['model_state_dict']
        else:
             state_dict = checkpoint # Assume checkpoint is the state dict itself
             
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing: logger.warning(f"Missing keys in state_dict: {missing}")
        if unexpected: logger.warning(f"Unexpected keys in state_dict: {unexpected}")
        
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
        return model, model_config # Return config too
        
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint {checkpoint_path}: {e}")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Sliding Window Inference on Landsat Scenes")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--scene_dir", required=True, help="Directory containing Landsat scene files (including *_MTL.txt and *_SR_B*.TIF)")
    parser.add_argument("--output_dir", required=True, help="Directory to save output predictions, confidence maps, and visualizations")
    # Optional arguments from config can be overridden here
    parser.add_argument("--batch_size", type=int, default=TESTING['sliding_window'].get('batch_size', 1), help="Batch size for processing windows")
    parser.add_argument("--patch_size", type=int, default=TESTING['sliding_window'].get('size', DATASET.get('patch_size', 256)), help="Patch size for sliding window")
    parser.add_argument("--stride", type=int, help="Stride for sliding window (default: patch_size // 2)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Override config from args if provided
    if args.patch_size: TESTING['sliding_window']['size'] = args.patch_size
    if args.stride:
         TESTING['sliding_window']['stride'] = args.stride
    else: # Set default stride based on potentially updated patch_size
        TESTING['sliding_window']['stride'] = TESTING['sliding_window']['size'] // 2
    if args.batch_size: TESTING['sliding_window']['batch_size'] = args.batch_size
    
    model, model_config = load_model(args.checkpoint, device)
    
    run_segmentation_inference(
        model=model,
        landsat_scene_path=args.scene_dir,
        output_dir=Path(args.output_dir),
        config=model_config,
        device=device,
        batch_size=TESTING['sliding_window']['batch_size'] # Use potentially updated batch size
    )

if __name__ == "__main__":
    main() 