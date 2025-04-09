from .model_config import ModelConfig
from .unet_model import UNetModel
import warnings

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig):
        """Creates the segmentation model based on the configuration."""
        if config.model_type.lower() == 'unet':
            print(f"Creating UNet model with backbone: {config.backbone_type}")
            return UNetModel(config)
        # Add other model types here if needed in the future
        # elif config.model_type.lower() == 'some_other_model':
        #     return SomeOtherModel(config)
        else:
            warnings.warn(f"Model type '{config.model_type}' not explicitly handled by factory. \n" \
                           "Falling back to UNetModel. Check config if this is unintended.")
            # Fallback to UNet if model_type is different but config might still be UNet-like
            return UNetModel(config)