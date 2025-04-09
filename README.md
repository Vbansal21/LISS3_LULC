# LISS3 Satellite Image Analysis Project

## Overview
This project focuses on the analysis and classification of LISS3 satellite imagery using advanced machine learning techniques. The project aims to develop a robust system for land use and land cover (LULC) classification, leveraging state-of-the-art deep learning models including CLIP and SAM (Segment Anything Model).

## Project Structure
```
LISS3_Project/
├── landsat_project/          # Main project code
│   ├── datasets/             # Dataset handling and processing
│   ├── models/              # Model architectures and implementations
│   ├── training/            # Training scripts and utilities
│   ├── inference/           # Inference and prediction scripts
│   ├── utils/               # Utility functions and helpers
│   ├── tests/               # Test cases and validation
│   ├── config.py            # Configuration settings
│   └── landsat_classification.py  # Main classification pipeline
├── data/                    # Raw and processed data
├── LISS3_output/            # Output results and visualizations
├── LULC_ManusAI/            # Additional LULC analysis tools
└── requirements.txt         # Project dependencies
```

## Features
- Advanced satellite image processing and analysis
- Integration of CLIP and SAM models for improved classification
- Custom training pipeline for LULC classification
- Efficient data handling and preprocessing
- Comprehensive testing and validation framework

## Dependencies
The project requires Python 3.x and the following key dependencies:
- PyTorch
- Rasterio
- NumPy
- OpenCV
- scikit-learn
- CLIP (from OpenAI)
- Segment Anything Model (SAM)
- Additional dependencies listed in requirements.txt

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd LISS3_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
The main functionality is contained within the `landsat_project` directory. To use the classification system:

1. Configure your settings in `landsat_project/config.py`
2. Run the classification pipeline:
```bash
python landsat_project/landsat_classification.py
```

## Data Organization
- Raw satellite imagery should be placed in the `data/` directory
- Processed outputs will be saved in `LISS3_output/`
- Model checkpoints and training artifacts are stored in `landsat_project/output/`

## Contributing
This project is currently in development. The main working code is contained within the `landsat_project` directory, which will be factored out into a separate repository in the future.

## License
[Specify your license here]

## Acknowledgments
- OpenAI for the CLIP model
- Meta AI for the Segment Anything Model
- [Add other acknowledgments as needed] 