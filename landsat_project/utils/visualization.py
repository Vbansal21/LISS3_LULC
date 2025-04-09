import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def get_default_lulc_colors():
    """Provides a default color map for LULC classes."""
    # Example colors - customize based on your specific classes
    # Ensure the number of colors matches your number of classes
    return [
        [0, 0, 0],        # 0: Background/Undefined
        [0, 100, 0],      # 1: Forest
        [0, 255, 0],      # 2: Grassland
        [255, 255, 0],    # 3: Cropland
        [165, 42, 42],    # 4: Built-up
        [0, 0, 255],      # 5: Water
        [255, 165, 0],    # 6: Barren
        # Add more colors as needed up to num_classes
    ]

def visualize_segmentation(segmentation_map, output_path, class_colors=None, title="Segmentation Map"):
    """
    Visualizes a segmentation map using specified class colors.

    Args:
        segmentation_map (np.ndarray): 2D array with integer class labels.
        output_path (str): Path to save the visualization image.
        class_colors (list, optional): List of RGB colors for each class. 
                                       Defaults to get_default_lulc_colors().
        title (str, optional): Title for the plot.
    """
    if class_colors is None:
        class_colors = get_default_lulc_colors()
    
    num_classes = len(class_colors)
    cmap = ListedColormap(np.array(class_colors) / 255.0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation_map, cmap=cmap, vmin=0, vmax=num_classes - 1)
    plt.title(title)
    plt.colorbar(ticks=np.arange(num_classes), label='Class ID')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Segmentation visualization saved to: {output_path}")

def visualize_change(map1, map2, output_path, class_colors=None, title="Change Detection Map"):
    """
    Visualizes the change between two segmentation maps.
    Pixels that changed class are highlighted.

    Args:
        map1 (np.ndarray): Segmentation map at time 1.
        map2 (np.ndarray): Segmentation map at time 2.
        output_path (str): Path to save the visualization image.
        class_colors (list, optional): List of RGB colors for each class. 
                                       Defaults to get_default_lulc_colors().
        title (str, optional): Title for the plot.
    """
    if map1.shape != map2.shape:
        raise ValueError("Input maps must have the same shape.")
    
    if class_colors is None:
        class_colors = get_default_lulc_colors()

    change_mask = (map1 != map2).astype(np.uint8)
    
    # Create a visualization showing changed pixels in red, unchanged in gray
    change_viz = np.zeros((*map1.shape, 3), dtype=np.uint8)
    change_viz[change_mask == 0] = [128, 128, 128]  # Gray for unchanged
    change_viz[change_mask == 1] = [255, 0, 0]      # Red for changed

    plt.figure(figsize=(10, 10))
    plt.imshow(change_viz)
    plt.title(title)
    plt.axis('off')

    # Add a simple legend (optional)
    handles = [
        plt.Rectangle((0,0),1,1, fc=[0.5, 0.5, 0.5]), # Gray
        plt.Rectangle((0,0),1,1, fc=[1.0, 0.0, 0.0])  # Red
    ]
    labels= ["Unchanged", "Changed"]
    plt.legend(handles, labels, loc='lower right')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Change visualization saved to: {output_path}") 