from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for UNet-based segmentation model"""
    # Core settings
    model_type: str = 'unet' # Keep for potential future model types, default to 'unet'
    num_classes: int = 11     # Default to ESA classes
    in_channels: int = 6      # Default to Landsat channels (RGB + NIR + SWIR1 + SWIR2)
    
    # Backbone settings
    backbone_type: str = 'efficientnet_b0' # Default backbone
    use_pretrained_backbone: bool = True
    freeze_backbone: bool = False
    
    # Decoder settings (UNet specific)
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16])
    use_skip_connections: bool = True
    
    # Other settings (can be extended)
    dropout_rate: float = 0.1
    
    # Special backbone settings
    sam_checkpoint_path: Optional[str] = None # Path to SAM checkpoint if using sam2
    clip_model_name: str = 'ViT-B/16'        # CLIP model variant
    
    # Deprecated/Removed fields:
    # - pretrained (moved to use_pretrained_backbone for clarity)
    # - classifier_type
    # - classifier_layers
    # - unet_encoder_channels (derived from backbone)
    # - model_specific_config (less generic now)

    def __post_init__(self):
        # Basic validation or warnings
        if self.backbone_type == 'sam2' and self.sam_checkpoint_path is None:
            print("Warning: SAM2 backbone selected but no checkpoint path provided in ModelConfig.sam_checkpoint_path")
        if self.model_type != 'unet':
            print(f"Warning: model_type is set to '{self.model_type}', but this config is primarily for UNet.") 