# User Guide: Deep Learning Model for Land Use/Land Cover Change and Wildfire Prediction

This user guide provides detailed instructions for using the deep learning model for Land Use/Land Cover Change prediction and wildfire prediction system.

## System Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Required Python packages:
  - numpy
  - matplotlib
  - scikit-learn
  - rasterio (for GeoTIFF handling)
  - torchvision
  - tqdm
  - PIL (Pillow)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-learning-landcover-wildfire.git
cd deep-learning-landcover-wildfire
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download pre-trained model checkpoints:
```bash
mkdir -p checkpoints
# Download model checkpoints from provided link or use your trained models
```

## Directory Structure

```
deep_learning_project/
├── data/                      # Data directory
│   ├── UCMerced_LandUse/      # UC Merced dataset
│   ├── ESA_WorldCover/        # ESA World Cover dataset
│   ├── Landsat/               # Landsat 8-9 imagery
│   └── processed/             # Processed datasets
├── models/                    # Model implementations
│   ├── sam2_unet.py           # Full SAM2-UNet model
│   └── sam2_unet_lite.py      # Lightweight version
├── scripts/                   # Utility scripts
│   └── data_preprocessing.py  # Data preprocessing utilities
├── checkpoints/               # Model checkpoints
├── results/                   # Evaluation results
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── predict.py                 # Prediction system
└── README.md                  # Project documentation
```

## Using the Prediction System

The prediction system allows you to apply the trained model to new data for land use/land cover classification, change detection, and wildfire prediction.

### Basic Usage

For basic prediction on a single image or directory of images:

```bash
python predict.py --task landcover --input-dir /path/to/images --output-dir /path/to/results --checkpoint checkpoints/landcover_best_model.pth
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Task to perform: 'landcover' or 'wildfire' | 'landcover' |
| `--input-dir` | Directory containing input images or a single image file | (Required) |
| `--output-dir` | Directory to save prediction results | 'predictions' |
| `--checkpoint` | Path to model checkpoint | (Required) |
| `--patch-size` | Size of image patches for processing | 128 |
| `--overlap` | Overlap between adjacent patches | 32 |
| `--batch-size` | Batch size for prediction | 4 |
| `--time-series` | Process input as time series for change detection | False |
| `--visualize` | Generate visualization of predictions | False |

### Example Use Cases

#### Land Cover Classification

To classify land cover in a set of aerial or satellite images:

```bash
python predict.py --task landcover --input-dir data/test_images --output-dir results/landcover --checkpoint checkpoints/landcover_best_model.pth --visualize
```

This will:
1. Load all images from `data/test_images`
2. Apply the land cover classification model
3. Save prediction maps to `results/landcover`
4. Generate visualizations of the predictions

#### Wildfire Detection

To detect wildfires in satellite imagery:

```bash
python predict.py --task wildfire --input-dir data/wildfire_images --output-dir results/wildfire --checkpoint checkpoints/wildfire_best_model.pth --visualize
```

#### Land Cover Change Detection

To detect changes in land cover between time series images:

```bash
python predict.py --task landcover --input-dir data/time_series --output-dir results/change_detection --checkpoint checkpoints/landcover_best_model.pth --time-series --visualize
```

For this to work correctly:
- Images in the input directory should be named in chronological order
- The system will process consecutive pairs of images to detect changes

### Input Data Requirements

#### Image Formats

The system supports the following image formats:
- GeoTIFF (.tif, .tiff)
- JPEG (.jpg, .jpeg)
- PNG (.png)

#### Image Size

There is no strict requirement for input image size, as the system will process images in patches. However, for optimal performance:
- Minimum recommended size: 128x128 pixels
- For very large images (>2000x2000 pixels), processing may take longer

#### Bands/Channels

- For land cover classification: RGB images (3 channels)
- For wildfire detection: Preferably RGB+NBR (4 channels), but RGB only is also supported

### Output Files

The prediction system generates the following outputs:

1. **Prediction Maps**:
   - `*_prediction.tif`: GeoTIFF files containing the predicted classes or probabilities
   - For land cover: Each pixel value corresponds to a land cover class
   - For wildfire: Each pixel value represents fire probability (0-255)

2. **Change Detection Maps** (when using `--time-series`):
   - `*_to_*_change.tif`: Maps showing detected changes between consecutive images
   - For land cover: Binary maps where 1 indicates change
   - For wildfire: Probability maps of fire spread

3. **Visualizations** (when using `--visualize`):
   - `*_visualization.png`: Side-by-side visualization of input and prediction
   - `*_to_*_change_visualization.png`: Visualization of change detection results

### Interpreting Results

#### Land Cover Classes

For land cover prediction, the pixel values in the output maps correspond to these classes:

| Class ID | Description |
|----------|-------------|
| 0 | Agricultural |
| 1 | Airplane |
| 2 | Baseball Diamond |
| 3 | Beach |
| 4 | Buildings |
| 5 | Chaparral |
| 6 | Dense Residential |
| 7 | Forest |
| 8 | Freeway |
| 9 | Golf Course |
| 10 | Harbor |
| 11 | Intersection |
| 12 | Medium Residential |
| 13 | Mobile Home Park |
| 14 | Overpass |
| 15 | Parking Lot |
| 16 | River |
| 17 | Runway |
| 18 | Sparse Residential |
| 19 | Storage Tanks |
| 20 | Tennis Court |

#### Wildfire Detection

For wildfire prediction, the output is a probability map:
- Values close to 0: Low probability of fire
- Values close to 255: High probability of fire

## Training Your Own Models

If you want to train your own models on custom data:

### Data Preparation

1. Organize your data following the structure in the `data` directory
2. Run the preprocessing script:
```bash
python scripts/data_preprocessing.py
```

### Training

Run the training script with your desired configuration:

```bash
python train.py --task landcover --data-dir /path/to/data --epochs 50 --batch-size 4 --lr 0.001 --patch-size 128
```

Key training parameters:
- `--task`: 'landcover' or 'wildfire'
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--patch-size`: Size of image patches for training

### Evaluation

Evaluate your trained model:

```bash
python evaluate.py --task landcover --data-dir /path/to/data --checkpoint /path/to/model.pth --output-dir /path/to/results
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Reduce patch size
   - Use the lightweight model version

2. **Missing GeoTIFF Support**:
   - Ensure rasterio is properly installed: `pip install rasterio`

3. **Poor Prediction Quality**:
   - Ensure input images are properly normalized
   - Check that you're using the correct model for your task
   - For wildfire detection, ensure NBR band is available if possible

4. **Slow Processing**:
   - Reduce patch size or increase overlap
   - Process smaller batches of images at a time
   - Use a machine with GPU support if available

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the project documentation in README.md
2. Look for similar issues in the project repository
3. Contact the project maintainers with detailed information about your problem

## Advanced Usage

### Custom Model Integration

You can integrate your own models by:
1. Creating a new model class in the `models` directory
2. Implementing the required forward method
3. Updating the model loading function in `predict.py`

### Processing Very Large Datasets

For processing large datasets efficiently:
1. Split your dataset into manageable chunks
2. Process each chunk separately
3. Use the `--batch-size` parameter to optimize memory usage

### Geospatial Analysis Integration

The prediction system preserves geospatial metadata when working with GeoTIFF files, allowing for:
1. Direct integration with GIS software
2. Spatial analysis of prediction results
3. Overlay with other geospatial datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.
