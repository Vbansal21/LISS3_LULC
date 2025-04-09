# Landsat LULC Classification & Change Detection

This project provides a framework for Land Use Land Cover (LULC) classification and change detection using Landsat satellite imagery, leveraging deep learning models trained in a two-stage process.

## Overview

The system utilizes a UNet-based architecture with configurable backbones (e.g., EfficientNet, ResNet) for semantic segmentation. It follows a two-stage training strategy:

1.  **Pre-training (Classification):** The model is optionally pre-trained on the UCMerced Land Use dataset for scene classification. This helps the backbone learn relevant visual features.
2.  **Fine-tuning (Segmentation):** The model is then fine-tuned on a dataset derived from ESA WorldCover and Landsat imagery for the primary task of LULC pixel-wise segmentation.

The framework supports:
*   Training the model through the two stages.
*   Running inference on new Landsat scenes to generate LULC maps.
*   Performing change detection between two Landsat scenes captured at different times.

## Features

*   **Model Architecture:** UNet with pluggable backbones (EfficientNet-B0, ResNet50 supported).
*   **Two-Stage Training:** UCMerced classification pre-training followed by ESA/Landsat segmentation fine-tuning.
*   **Datasets:** Integrated support for UCMerced and ESA WorldCover/Landsat data.
*   **Inference:** Predict LULC maps for full Landsat scenes.
*   **Change Detection:** Identify areas of LULC change between two time points.
*   **Configuration:** Flexible settings via `config.py` and command-line arguments.
*   **Monitoring:** Weights & Biases integration for experiment tracking and visualization.
*   **Checkpointing:** Saves the best model based on validation metrics.
*   **Visualization:** Generates visual outputs for segmentation results during validation and inference.

## Project Structure

```
landsat_project/
├── README.md             # This file
├── landsat_classification.py # Main executable script
├── config.py             # Configuration settings (paths, model params, training params)
├── requirements.txt      # Python dependencies for this sub-project
├── setup.py              # For package installation (if needed)
├── __init__.py
│
├── datasets/             # Data loading and preprocessing logic
│   ├── __init__.py
│   ├── esa_dataset.py      # ESA WorldCover / Landsat dataset loader
│   └── ucmerced_dataset.py # UCMerced dataset loader
│
├── models/               # Model definitions
│   ├── __init__.py
│   ├── model_config.py     # Dataclass for model configuration
│   ├── model_factory.py    # Factory to create models based on config
│   └── unet_model.py       # UNet implementation and backbone feature extraction
│
├── training/             # (Contains training loop logic - integrated into main script)
│
├── inference/            # Inference and change detection logic
│   ├── __init__.py
│   ├── change_detection.py # Logic for LULC change detection
│   └── inference.py        # Logic for running inference on scenes
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── logging.py          # Logging setup
│   └── metrics.py          # Segmentation metrics calculation
│
├── LULC_CLIP/            # (Sub-module, potentially related to CLIP integration)
├── data/                 # Default location for storing datasets (can be configured)
├── output/               # Default location for saving models, logs, visualizations (can be configured)
├── logs/                 # Log file directory (within output/)
├── tests/                # Unit tests (if any)
└── wandb/                # Weights & Biases local artifacts
```

## Installation

1.  **Clone the repository:**
    ```bash
    # Navigate to the parent LISS3_Project directory if not already there
    # git clone ... (if you haven't already)
    cd LISS3_Project
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    Install dependencies from the *main* `requirements.txt` in the `LISS3_Project` root directory, as it should contain all necessary packages for this sub-project as well.
    ```bash
    pip install -r requirements.txt
    ```
    *(Alternatively, if `landsat_project/requirements.txt` is maintained separately and you want to install this as a package)*:
    ```bash
    # cd landsat_project
    # pip install -e .
    # cd ..
    ```

## Configuration (`config.py`)

The `config.py` file centralizes most settings. Key sections:

*   **`PATHS`**: Defines paths to datasets (UCMerced, ESA WorldCover, Landsat) and output directories. **Update these paths** to match your system.
*   **`MODEL`**: Configures the model architecture:
    *   `model_type`: e.g., 'unet'
    *   `backbone_type`: e.g., 'efficientnet-b0'
    *   `use_pretrained_backbone`: Boolean flag.
    *   `num_classes`: Dictionary mapping dataset names ('ucmerced', 'esa') to the number of classes.
    *   `in_channels`: Dictionary mapping dataset names to input channels (e.g., 3 for RGB UCMerced, 4+ for Landsat bands).
    *   `decoder_channels`: List of channel sizes for the UNet decoder.
*   **`TRAINING`**: Configures the training process:
    *   `device`: 'cuda' or 'cpu'.
    *   `num_epochs`: Dictionary mapping dataset names to training epochs.
    *   `batch_size`: Dictionary mapping dataset names to batch sizes.
    *   `optimizer`: Settings for the AdamW optimizer (lr, weight_decay).
    *   `scheduler`: Settings for the OneCycleLR scheduler.
    *   `num_workers`: Number of workers for DataLoader.
*   **`DATASET`**: Dataset-specific parameters:
    *   `patch_size`: Input image size.
    *   `normalization`: Dictionary with mean/std values for different datasets.
    *   `ignore_index`: Pixel value to ignore during loss calculation.
    *   `num_samples`: Number of samples to use from ESA/Landsat dataset (useful for debugging/testing).
*   **`LOGGING`**: Settings for console/file logging and Weights & Biases.
*   **`OUTPUT`**: Output directory configuration.
*   **`VISUALIZATION`**: Settings for result visualization (frequency, class colors).

## Data Setup

1.  **UCMerced Land Use Dataset:**
    *   Download the dataset.
    *   Extract it to the path specified in `PATHS['uc_merced']` in `config.py`.
    *   The directory should contain subdirectories for each class (e.g., `agricultural`, `airplane`, ...).

2.  **ESA WorldCover & Landsat Data:**
    *   **ESA WorldCover:** Download the relevant tiles (e.g., from Google Earth Engine or other sources). Place them in the directory specified by `PATHS['esa_worldcover']`. The code might expect a specific structure within this directory.
    *   **Landsat Data:** Obtain Landsat scenes (e.g., Collection 2 Level-2 Surface Reflectance from USGS EarthExplorer). Place the scene directories (containing the individual band TIF files like `B1.TIF`, `B2.TIF`, ...) in the directory specified by `PATHS['landsat']`.
    *   The `esa_dataset.py` script handles pairing Landsat patches with corresponding ESA labels. Ensure the paths and data organization align with its expectations.

## Usage (`landsat_classification.py`)

The main script `landsat_classification.py` orchestrates the different modes.

**Modes:**

*   `--mode train_inference` (Default): Runs the full pipeline:
    1.  Trains on UCMerced (if `ucmerced_epochs` > 0).
    2.  Trains/Fine-tunes on ESA/Landsat (if `esa_epochs` > 0), potentially loading weights from the UCMerced stage.
    3.  Performs inference on the Landsat scene specified by `--landsat_path_t1`, using the best checkpoint found (prioritizing ESA, then UCMerced, then user-provided).
*   `--mode change_detect`: Performs LULC change detection between two Landsat scenes. Requires a trained model checkpoint.

**Common Workflow:**

1.  **Configure `config.py`:** Set correct paths and desired parameters.
2.  **Prepare Data:** Ensure datasets are downloaded and placed correctly.
3.  **Run Training & Inference:**
    ```bash
    python landsat_project/landsat_classification.py --mode train_inference \
           --output_dir path/to/your/experiment/output \
           --ucmerced_epochs 50 \
           --esa_epochs 100 \
           --landsat_path_t1 path/to/your/target/landsat_scene_for_inference
           # Add other arguments as needed (e.g., --batch_size, --lr)
    ```
    *   This will train both stages and then run inference on `landsat_path_t1`.
    *   Model checkpoints (`ucmerced_ucmerced_best_model.pth`, `esa_esa_best_model.pth`) will be saved in `path/to/your/experiment/output/models/`.
    *   Inference results will be in `path/to/your/experiment/output/inference/`.

4.  **Run Change Detection (Optional):**
    ```bash
    python landsat_project/landsat_classification.py --mode change_detect \
           --checkpoint_path path/to/your/experiment/output/models/esa/esa_esa_best_model.pth \
           --landsat_path_t1 path/to/first/landsat_scene \
           --landsat_path_t2 path/to/second/landsat_scene \
           --output_dir path/to/your/change_detection/output
    ```
    *   Requires a trained checkpoint (`--checkpoint_path`).
    *   Outputs a change map and summary JSON to the specified output directory.

**Command Line Arguments:**

Run `python landsat_project/landsat_classification.py --help` for a full list. Key arguments:

*   `--mode`: `train_inference` or `change_detect`.
*   `--model_type`: Override model type from config (e.g., 'unet').
*   `--backbone_type`: Override backbone from config (e.g., 'efficientnet-b0').
*   `--landsat_path_t1`: Path to Landsat scene directory (for inference or first scene in change detection).
*   `--landsat_path_t2`: Path to second Landsat scene directory (for change detection).
*   `--output_dir`: Base directory for saving all outputs (models, logs, inference results). *Default from `config.py`*.
*   `--checkpoint_path`: Path to a specific model checkpoint to load. Used in `change_detect` mode or to override automatic checkpoint selection in `train_inference`.
*   `--test_mode`: Run with fewer samples (useful for debugging).
*   `--pretrained`: Boolean flag to force override usage of pretrained backbone weights (ignores config setting).
*   `--no_wandb`: Disable Weights & Biases logging.
*   `--ucmerced_epochs`: Override number of UCMerced training epochs. Set to 0 to skip.
*   `--esa_epochs`: Override number of ESA training epochs. Set to 0 to skip.
*   `--batch_size`: Override batch size for all stages.
*   `--lr`: Override initial learning rate.
*   `--load_ucmerced_from`: Explicitly specify a UCMerced checkpoint to load before ESA training (overrides automatic loading).

## Output Structure

Outputs are saved relative to the specified `--output_dir`.

*   `models/`: Contains saved model checkpoints.
    *   `ucmerced/ucmerced_ucmerced_best_model.pth`
    *   `esa/esa_esa_best_model.pth`
*   `visualizations/`: Image samples saved during validation (if enabled).
    *   `ucmerced/`
    *   `esa/`
*   `logs/`: Log files (e.g., `training.log`).
*   `inference/`: Results from inference runs.
    *   `[scene_id]/`: Subdirectory for each inferred scene.
        *   `classification.tif`: Predicted LULC map.
        *   `confidence.tif`: Confidence map (e.g., max softmax probability).
        *   `visualization.png`: Visual comparison of input/prediction.
*   `change_detection/`: Results from change detection runs.
    *   `change_map.tif`: Map highlighting areas of change.
    *   `change_summary.json`: JSON file summarizing changes.

## License

MIT License (based on the original file, please confirm/update if needed) 