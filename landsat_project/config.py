import os
from pathlib import Path
import torch
from datetime import datetime

# =============================================================================
# Paths Configuration
# =============================================================================
# Assume script is run from within landsat_project directory
PATHS = {
    'uc_merced': Path("./data/UCMerced_LandUse/UCMerced_LandUse/"), # Relative to landsat_project
    'esa_worldcover': Path("./data/ESA_WorldCover_10m_2021_v200_N30W120/"), # Relative to landsat_project
    'landsat': Path("./data/raw/Landsat/"), # Relative to landsat_project
    'shapefile': Path("./data/raw/Landsat/landsat_ot_c2_l2_67f39a374ef6117b.shp"), # Relative to landsat_project
    'output': Path("./output"), # Relative to landsat_project
    'logs': Path("./logs"),  # Relative to landsat_project
}

# Create necessary directories (skip files)
for path_name, path in PATHS.items():
    if path_name not in ['shapefile']:  # Skip file paths
        path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================
MODEL = {
    'model_type': 'unet', # Default to UNet
    'num_classes': {
        'ucmerced': 21,  # UCMerced land use classes
        'esa': 11       # ESA WorldCover classes (default)
    },
    'in_channels': {
        'ucmerced': 3,  # RGB
        'esa': 6        # 6 bands for Landsat SR (B2-B7)
    },
    # Supported: resnet18, resnet34, resnet50, resnet101, resnet152,
    #            efficientnet-b0, efficientnet-b1, ..., efficientnet-b7,
    #            inception_v3,
    #            clip:ViT-B/32, clip:ViT-B/16, clip:ViT-L/14, ...
    #            sam2 (requires sam_checkpoint_path in ModelConfig)
    'backbone_type': 'efficientnet-b0', # Default backbone
    'use_pretrained_backbone': True,
    'freeze_backbone': False,
    'decoder_channels': [128, 64, 32, 16], # Channels for UNet decoder (4 stages for EfficientNet-b0's 5 feature maps)
    'use_skip_connections': True,
    'dropout_rate': 0.1,
    
    # Specific configs (add if needed)
    # 'sam_checkpoint_path': '/path/to/sam_checkpoint.pth',
    # 'clip_model_name': 'ViT-L/14' 
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'batch_size': {
        'ucmerced': 16,  # Smaller batch size for higher resolution images
        'esa': 32,      # Larger batch size for ESA data
    },
    'num_epochs': {
        'ucmerced': 50,  # Train longer on UCMerced
        'esa': 30,      # Fine-tune on ESA
    },
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
    },
    'scheduler': {
        'type': 'one_cycle',
        'max_lr': 1e-3,
        'pct_start': 0.3,
        'div_factor': 25.0,
        'final_div_factor': 1e4,
    },
    'num_workers': 4,
    'pin_memory': True,
}

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET = {
    'patch_size': 224,
    'num_samples': 20000,
    'ignore_index': 255,  # For pixels to ignore in loss calculation
    'normalization': {
        'ucmerced': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean for RGB
            'std': [0.229, 0.224, 0.225],   # ImageNet std for RGB
        },
        'esa': {
            'mean': [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # ImageNet mean duplicated for B2-B7
            'std': [0.229, 0.224, 0.225, 0.229, 0.224, 0.225],   # ImageNet std duplicated for B2-B7
        }
    },
    'augmentation': {
        'geometric': {
            'rotate': [-180, 180],
            'scale': [0.8, 1.2],
            'horizontal_flip': True,
            'vertical_flip': True,
        },
        'color': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
        },
        'noise': {
            'gaussian_noise': 0.02,
            'gaussian_blur': 0.5,
        }
    },
    'overlap_threshold': 0.5,  # Minimum overlap required for ESA-Landsat matching
    'max_cloud_coverage': 0.1, # Maximum allowed cloud coverage
}

# =============================================================================
# Logging Configuration
# =============================================================================
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': str(PATHS['logs']),  # Use the logs path from PATHS
    'log_file': str(PATHS['logs'] / 'landsat_segmentation.log'),  # Use Path for joining
    'experiment_name': 'landsat_segmentation',
    'wandb': {
        'project': 'landsat-segmentation',
        'entity': None,  # Your wandb username/organization
        'log_artifacts': True,
        'log_model': True,
        'log_code': True,
    },
    'tensorboard': {
        'enabled': True,
        'log_dir': str(PATHS['logs'] / 'tensorboard'),  # Use Path for joining
    }
}

# =============================================================================
# Output Configuration
# =============================================================================
OUTPUT = {
    'output_dir': Path('./output'),
    'model_dir': Path('./output/models'),
    'visualization_dir': Path('./output/visualizations'),
    'metrics_dir': Path('./output/metrics'),
    'save_frequency': 5,  # Save checkpoints every N epochs
    'keep_last_n': 3,    # Keep only last N checkpoints
    'save_best': True,   # Save best model based on validation loss
    'save_last': True,   # Save last model of training
}

# Create necessary directories
for dir_path in [OUTPUT['output_dir'], OUTPUT['model_dir'], 
                OUTPUT['visualization_dir'], OUTPUT['metrics_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ensure logs directory exists (from PATHS)
PATHS['logs'].mkdir(parents=True, exist_ok=True)

# =============================================================================
# Testing Configuration
# =============================================================================
TESTING = {
    'metrics': {
        'iou': True,          # Intersection over Union
        'dice': True,         # Dice coefficient
        'accuracy': True,     # Pixel accuracy
        'precision': True,    # Precision per class
        'recall': True,       # Recall per class
        'f1': True,          # F1 score per class
        'confusion_matrix': True,
    },
    'visualization': {
        'save_predictions': True,
        'save_uncertainty': True,
        'save_attention_maps': False,
        'overlay_predictions': True,
    },
    'sliding_window': {
        'size': 224,
        'stride': 112,
        'batch_size': 16,
    }
}

# =============================================================================
# Visualization Configuration
# =============================================================================
VISUALIZATION = {
    'class_colors': {
        'ucmerced': {
            0: (0, 255, 0),     # Agricultural
            1: (128, 0, 0),     # Airplane
            2: (0, 0, 255),     # Baseball Diamond
            # ... add colors for all UCMerced classes
        },
        'esa': {
            10: (0, 100, 0),     # Tree cover
            20: (150, 200, 0),   # Shrubland
            30: (255, 255, 0),   # Grassland
            40: (255, 180, 0),   # Cropland
            50: (255, 0, 0),     # Built-up
            60: (210, 210, 210), # Bare
            70: (255, 255, 255), # Snow and ice
            80: (0, 0, 255),     # Water
            90: (0, 200, 200),   # Wetland
            95: (0, 150, 150),   # Mangroves
            100: (200, 200, 0)   # Moss and lichen
        }
    },
    'dpi': 300,
    'figsize': (12, 8),
    'save_predictions': True,
    'save_uncertainty': True,
    'colormap': 'nipy_spectral'
} 