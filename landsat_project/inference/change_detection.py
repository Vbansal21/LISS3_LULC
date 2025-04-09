import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import os
from pathlib import Path

# Add project root to allow imports from utils
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.visualization import visualize_segmentation, visualize_change
from utils.metrics import calculate_change_metrics

# --- Helper Functions (Assume these exist or adapt as needed) ---

def preprocess_landsat_patch(patch):
    """Placeholder: Preprocess a Landsat patch for the model."""
    # Apply normalization, channel selection/reordering, etc.
    # Example: Normalize to [0, 1]
    patch = patch.astype(np.float32) / np.iinfo(patch.dtype).max
    # Convert to PyTorch tensor (C, H, W)
    return torch.from_numpy(patch).float()

def postprocess_output(output_tensor):
    """Placeholder: Postprocess model output to get class map."""
    # Example: Apply argmax to get class indices
    return torch.argmax(output_tensor, dim=1).cpu().numpy()

# ---------------------------------------------------------------

def run_inference_on_raster(model, device, raster_path, output_map_path, batch_size=16, patch_size=256):
    """
    Runs inference on a large raster file using a sliding window approach.

    Args:
        model: Trained PyTorch segmentation model.
        device: CUDA device or CPU.
        raster_path (str): Path to the input raster file (e.g., Landsat GeoTIFF).
        output_map_path (str): Path to save the output segmentation map (GeoTIFF).
        batch_size (int): Number of patches to process in parallel.
        patch_size (int): Size of the square patches for inference.

    Returns:
        str: Path to the saved segmentation map.
    """
    model.eval()
    
    with rasterio.open(raster_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, nodata=0) # Output class map
        
        rows, cols = src.height, src.width
        output_map = np.zeros((rows, cols), dtype=np.uint8)
        
        patches = []
        coords = []

        print(f"Processing {raster_path}...")
        with torch.no_grad():
            for r in tqdm(range(0, rows, patch_size), desc="Rows"):
                for c in range(0, cols, patch_size):
                    h = min(patch_size, rows - r)
                    w = min(patch_size, cols - c)
                    window = Window(c, r, w, h)
                    
                    patch_data = src.read(window=window)
                    # Handle cases where patch is smaller than patch_size (pad?)
                    # Basic check: skip if too small or handle padding
                    if patch_data.shape[1] < patch_size or patch_data.shape[2] < patch_size:
                        # Simple approach: skip partial edge patches or implement padding
                        print(f"Skipping/Padding edge patch at ({r},{c}) size ({h},{w})")
                        # Or pad here: patch_data = np.pad(...) 
                        continue 

                    processed_patch = preprocess_landsat_patch(patch_data).unsqueeze(0) # Add batch dim
                    patches.append(processed_patch)
                    coords.append((r, c, h, w))
                    
                    if len(patches) == batch_size:
                        batch_tensor = torch.cat(patches, dim=0).to(device)
                        outputs = model(batch_tensor)
                        pred_maps = postprocess_output(outputs) # Shape: (B, H, W)
                        
                        for i, (pr, pc, ph, pw) in enumerate(coords):
                            output_map[pr:pr+ph, pc:pc+pw] = pred_maps[i, :ph, :pw]
                        
                        patches.clear()
                        coords.clear()

            # Process any remaining patches
            if patches:
                batch_tensor = torch.cat(patches, dim=0).to(device)
                outputs = model(batch_tensor)
                pred_maps = postprocess_output(outputs)
                
                for i, (pr, pc, ph, pw) in enumerate(coords):
                     output_map[pr:pr+ph, pc:pc+pw] = pred_maps[i, :ph, :pw]

        print(f"Saving segmentation map to {output_map_path}...")
        os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
        with rasterio.open(output_map_path, 'w', **profile) as dst:
            dst.write(output_map, 1)
            
    return output_map_path

def detect_lulc_change(model, device, landsat_path_t1, landsat_path_t2, output_dir, num_classes, config):
    """
    Performs LULC inference on two Landsat images and detects changes.

    Args:
        model: Trained PyTorch segmentation model.
        device: CUDA device or CPU.
        landsat_path_t1 (str): Path to Landsat image at time 1.
        landsat_path_t2 (str): Path to Landsat image at time 2.
        output_dir (str): Directory to save results (maps, visualizations, metrics).
        num_classes (int): Number of LULC classes.
        config: Configuration object (e.g., from config file) potentially containing 
                visualization settings like class_colors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Inference --- 
    print("--- Running Inference Time 1 ---")
    map_path_t1 = os.path.join(output_dir, "segmentation_map_t1.tif")
    run_inference_on_raster(model, device, landsat_path_t1, map_path_t1)

    print("--- Running Inference Time 2 ---")
    map_path_t2 = os.path.join(output_dir, "segmentation_map_t2.tif")
    run_inference_on_raster(model, device, landsat_path_t2, map_path_t2)

    # --- Load Results --- 
    with rasterio.open(map_path_t1) as src:
        seg_map_t1 = src.read(1)
    with rasterio.open(map_path_t2) as src:
        seg_map_t2 = src.read(1)

    # --- Visualization --- 
    print("--- Generating Visualizations ---")
    class_colors = config.VISUALIZATION.get('CLASS_COLORS', None) # Get colors from config if available
    viz_path_t1 = os.path.join(output_dir, "segmentation_map_t1.png")
    visualize_segmentation(seg_map_t1, viz_path_t1, class_colors=class_colors, title="Segmentation Map Time 1")

    viz_path_t2 = os.path.join(output_dir, "segmentation_map_t2.png")
    visualize_segmentation(seg_map_t2, viz_path_t2, class_colors=class_colors, title="Segmentation Map Time 2")

    change_viz_path = os.path.join(output_dir, "change_map.png")
    visualize_change(seg_map_t1, seg_map_t2, change_viz_path, class_colors=class_colors, title="Change Detection (T1 vs T2)")

    # --- Metrics --- 
    print("--- Calculating Change Metrics ---")
    change_metrics = calculate_change_metrics(seg_map_t1, seg_map_t2, num_classes)
    
    print("\nChange Metrics:")
    print(f"  Total Pixels: {change_metrics['Total Pixels']}")
    print(f"  Changed Pixels: {change_metrics['Changed Pixels']}")
    print(f"  Percentage Change: {change_metrics['Percentage Change']:.2f}%")
    print("\nTransition Matrix (Rows=T1, Cols=T2):")
    print(change_metrics['Transition Matrix'])
    
    # Save metrics to a file
    metrics_path = os.path.join(output_dir, "change_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Change Metrics:\n")
        f.write(f"  Total Pixels: {change_metrics['Total Pixels']}\n")
        f.write(f"  Changed Pixels: {change_metrics['Changed Pixels']}\n")
        f.write(f"  Percentage Change: {change_metrics['Percentage Change']:.2f}%\n")
        f.write("\nTransition Matrix (Rows=T1, Cols=T2):\n")
        f.write(change_metrics['Transition Matrix'].to_string())
    print(f"Change metrics saved to: {metrics_path}")

    print("--- Change Detection Complete ---") 