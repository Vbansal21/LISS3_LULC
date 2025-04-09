import os
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import rasterio
import json

from landsat_project.inference.inference import (
    load_model,
    preprocess_landsat_swath,
    run_inference,
    extract_landsat_metadata
)

@pytest.fixture
def sample_landsat_swath():
    # Create a dummy Landsat-like swath with 7 bands
    image = np.random.rand(7, 256, 256).astype(np.float32)  # 7 bands (Landsat 8)
    return image

@pytest.fixture
def landsat_filename():
    return "LC08_L1TP_123032_20200101_20200101_01_RT.TIF"

def test_extract_landsat_metadata(landsat_filename):
    metadata = extract_landsat_metadata(landsat_filename)
    assert metadata['sensor'] == 'LC08'
    assert metadata['processing_level'] == 'L1TP'
    assert metadata['path'] == '123'
    assert metadata['row'] == '032'
    assert metadata['acquisition_date'] == '20200101'

def test_preprocess_landsat_swath(sample_landsat_swath, landsat_filename):
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
        # Save sample image to temporary file
        with rasterio.open(tmp.name, 'w', 
                         driver='GTiff',
                         height=sample_landsat_swath.shape[1],
                         width=sample_landsat_swath.shape[2],
                         count=sample_landsat_swath.shape[0],
                         dtype=sample_landsat_swath.dtype) as dst:
            dst.write(sample_landsat_swath)
        
        # Test preprocessing
        image, profile, metadata = preprocess_landsat_swath(tmp.name)
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 1  # Batch dimension
        assert image.shape[1] == sample_landsat_swath.shape[0]  # Number of bands
        assert isinstance(profile, dict)
        assert isinstance(metadata, dict)
        assert 'sensor' in metadata

def test_run_inference_with_metadata(sample_landsat_swath, landsat_filename):
    device = torch.device("cpu")
    model = load_model("dummy_checkpoint.pt", device)
    
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(suffix='.tif') as input_tmp, \
         tempfile.NamedTemporaryFile(suffix='.tif') as output_tmp, \
         tempfile.NamedTemporaryFile(suffix='.json') as metadata_tmp:
        
        # Save sample image to temporary input file
        with rasterio.open(input_tmp.name, 'w', 
                         driver='GTiff',
                         height=sample_landsat_swath.shape[1],
                         width=sample_landsat_swath.shape[2],
                         count=sample_landsat_swath.shape[0],
                         dtype=sample_landsat_swath.dtype) as dst:
            dst.write(sample_landsat_swath)
        
        # Run inference with metadata saving
        run_inference(model, input_tmp.name, output_tmp.name, device, save_metadata=True)
        
        # Verify output files exist
        assert os.path.exists(output_tmp.name)
        metadata_path = str(Path(output_tmp.name).with_suffix('.json'))
        assert os.path.exists(metadata_path)
        
        # Verify output properties
        with rasterio.open(output_tmp.name) as src:
            assert src.count == 1  # Single band output
            assert src.dtype == np.uint8
        
        # Verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
            assert 'sensor' in metadata
            assert 'path' in metadata
            assert 'row' in metadata 