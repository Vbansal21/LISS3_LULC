import os
import glob
import zipfile
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from rasterio.mask import mask
from shapely.geometry import box
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LandsatPreprocessor:
    """Class to handle Landsat data preprocessing including unzipping, metadata extraction,
    and data preparation for model input."""
    
    def __init__(self, 
                 landsat_dir: str,
                 shapefile_path: str,
                 output_dir: str,
                 bands: List[str] = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                 resolution: int = 30):
        """
        Initialize the Landsat preprocessor.
        
        Args:
            landsat_dir: Directory containing Landsat zip files
            shapefile_path: Path to the shapefile for clipping
            output_dir: Directory to save processed data
            bands: List of bands to process
            resolution: Target resolution in meters
        """
        self.landsat_dir = Path(landsat_dir)
        self.shapefile_path = Path(shapefile_path)
        self.output_dir = Path(output_dir)
        self.bands = bands
        self.resolution = resolution
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load shapefile
        self.aoi = gpd.read_file(self.shapefile_path)
        
        # Initialize metadata storage
        self.metadata_df = pd.DataFrame()

    def unzip_scenes(self) -> List[Path]:
        """
        Unzip all Landsat scenes in the input directory.
        
        Returns:
            List of paths to unzipped scene directories
        """
        scene_dirs = []
        zip_files = list(self.landsat_dir.glob("*.zip"))
        
        for zip_file in zip_files:
            scene_dir = self.landsat_dir / zip_file.stem
            if not scene_dir.exists():
                logger.info(f"Unzipping {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(scene_dir)
            scene_dirs.append(scene_dir)
        
        return scene_dirs

    def extract_metadata(self, scene_dir: Path) -> Dict:
        """
        Extract metadata from Landsat MTL file.
        
        Args:
            scene_dir: Path to unzipped scene directory
            
        Returns:
            Dictionary containing metadata
        """
        mtl_file = list(scene_dir.glob("*MTL.txt"))[0]
        metadata = {}
        
        with open(mtl_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if '=' in line:
                key, value = line.strip().split(' = ')
                metadata[key.strip()] = value.strip().strip('"')
        
        # Extract essential metadata
        metadata = {
            'scene_id': scene_dir.name,
            'acquisition_date': metadata.get('DATE_ACQUIRED', ''),
            'cloud_cover': float(metadata.get('CLOUD_COVER', 0)),
            'sun_elevation': float(metadata.get('SUN_ELEVATION', 0)),
            'sun_azimuth': float(metadata.get('SUN_AZIMUTH', 0)),
            'spacecraft': metadata.get('SPACECRAFT_ID', ''),
            'processing_level': metadata.get('PROCESSING_LEVEL', '')
        }
        
        return metadata

    def preprocess_scene(self, scene_dir: Path) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess a single Landsat scene.
        
        Args:
            scene_dir: Path to unzipped scene directory
            
        Returns:
            Tuple of (preprocessed array, metadata)
        """
        # Extract metadata
        metadata = self.extract_metadata(scene_dir)
        
        # Read and stack bands
        band_arrays = []
        for band in self.bands:
            band_file = list(scene_dir.glob(f"*_{band}.TIF"))[0]
            with rasterio.open(band_file) as src:
                # Get the geometry in the CRS of the raster
                aoi_transformed = self.aoi.to_crs(src.crs)
                
                # Perform the clipping
                clipped_array, clipped_transform = mask(src, 
                                                      aoi_transformed.geometry,
                                                      crop=True)
                
                # Add to list of bands
                band_arrays.append(clipped_array[0])
        
        # Stack bands
        stacked_array = np.stack(band_arrays)
        
        # Add spatial reference info to metadata
        with rasterio.open(list(scene_dir.glob(f"*_{self.bands[0]}.TIF"))[0]) as src:
            metadata.update({
                'crs': src.crs.to_string(),
                'transform': clipped_transform.to_gdal(),
                'width': stacked_array.shape[2],
                'height': stacked_array.shape[1]
            })
        
        return stacked_array, metadata

    def save_processed_data(self, 
                          array: np.ndarray, 
                          metadata: Dict, 
                          output_name: str):
        """
        Save processed data and metadata.
        
        Args:
            array: Preprocessed array
            metadata: Associated metadata
            output_name: Name for output files
        """
        # Save array
        output_path = self.output_dir / f"{output_name}.npy"
        np.save(output_path, array)
        
        # Save metadata
        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        pd.Series(metadata).to_json(metadata_path)
        
        # Update metadata DataFrame
        self.metadata_df = pd.concat([
            self.metadata_df,
            pd.DataFrame([metadata])
        ])

    def process_all_scenes(self):
        """Process all Landsat scenes in the input directory."""
        # Unzip all scenes
        scene_dirs = self.unzip_scenes()
        
        # Process each scene
        for scene_dir in scene_dirs:
            try:
                logger.info(f"Processing scene: {scene_dir.name}")
                
                # Skip if already processed
                output_name = scene_dir.name
                if (self.output_dir / f"{output_name}.npy").exists():
                    logger.info(f"Scene {output_name} already processed, skipping...")
                    continue
                
                # Preprocess scene
                array, metadata = self.preprocess_scene(scene_dir)
                
                # Save processed data
                self.save_processed_data(array, metadata, output_name)
                
            except Exception as e:
                logger.error(f"Error processing scene {scene_dir.name}: {str(e)}")
                continue
        
        # Save complete metadata
        if not self.metadata_df.empty:
            self.metadata_df.to_csv(self.output_dir / "scenes_metadata.csv", index=False)

def main():
    """Example usage of the LandsatPreprocessor class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Landsat scenes")
    parser.add_argument("--landsat_dir", required=True, help="Directory containing Landsat zip files")
    parser.add_argument("--shapefile_path", required=True, help="Path to shapefile for clipping")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed data")
    parser.add_argument("--bands", nargs="+", default=['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                        help="List of bands to process")
    parser.add_argument("--resolution", type=int, default=30,
                        help="Target resolution in meters")
    
    args = parser.parse_args()
    
    # Initialize and run preprocessor
    preprocessor = LandsatPreprocessor(
        landsat_dir=args.landsat_dir,
        shapefile_path=args.shapefile_path,
        output_dir=args.output_dir,
        bands=args.bands,
        resolution=args.resolution
    )
    
    preprocessor.process_all_scenes()

if __name__ == "__main__":
    main() 