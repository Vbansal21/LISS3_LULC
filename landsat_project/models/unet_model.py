import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Optional
import warnings

try:
    from efficientnet_pytorch import EfficientNet
    _has_efficientnet = True
except ImportError:
    _has_efficientnet = False
    EfficientNet = None

try:
    import timm
    _has_timm = True
except ImportError:
    _has_timm = False
    timm = None

try:
    import clip
    _has_clip = True
except ImportError:
    _has_clip = False
    clip = None

# Placeholder for SAM - replace with actual import if available
# try:
#     from segment_anything import sam_model_registry
#     _has_sam = True
# except ImportError:
#     _has_sam = False
_has_sam = False # Assuming SAM is not installed/integrated yet

from .model_config import ModelConfig

# Helper function to get intermediate layer outputs for skip connections
# (Specific implementation depends heavily on the backbone architecture)
# This is a simplified example; a more robust solution might use hooks
def get_intermediate_features(backbone_type: str, backbone: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Extract intermediate features for skip connections (highly model-dependent)."""
    features = []
    backbone_type = backbone_type.lower() # Ensure lowercase for consistent checks

    if backbone_type.startswith('resnet'):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        features.append(x) # After initial block (stem)
        x = backbone.maxpool(x)
        x = backbone.layer1(x); features.append(x)
        x = backbone.layer2(x); features.append(x)
        x = backbone.layer3(x); features.append(x)
        x = backbone.layer4(x); features.append(x) # Final encoder layer output
        # Return in order [stem_out, layer1_out, layer2_out, layer3_out, layer4_out]
        
    elif backbone_type.startswith('efficientnet') and _has_efficientnet:
        # EfficientNet feature extraction using endpoints
        # Assumes backbone has an `extract_endpoints` method (common in efficientnet_pytorch)
        try:
            endpoints = backbone.extract_endpoints(x)
            # Example endpoints keys for b0 (might differ for others, inspect `backbone.extract_endpoints` output)
            # Typically named like 'reduction_1', 'reduction_2', ... 'reduction_5' or similar
            # Need a reliable mapping from backbone_type to endpoint keys
            endpoint_keys = [f'reduction_{i}' for i in range(1, 6)] # Placeholder keys
            features = [endpoints[key] for key in endpoint_keys if key in endpoints]
            if not features:
                 warnings.warn(f"Could not extract expected endpoints for {backbone_type}. Check keys.")
                 # Fallback: Use approximate feature extraction
                 x = backbone._swish(backbone._bn0(backbone._conv_stem(x)))
                 # Approximate indices where reduction occurs (highly unreliable)
                 reduction_indices = [i for i, block in enumerate(backbone._blocks) if block._depthwise_conv.stride == [2, 2]]
                 reduction_indices = [0] + reduction_indices + [len(backbone._blocks)-1]
                 current_feat = x
                 temp_features = []
                 for i, block in enumerate(backbone._blocks):
                     current_feat = block(current_feat)
                     if i in reduction_indices : # Just an example, needs proper indices
                          temp_features.append(current_feat)
                 # Final feature map
                 x = backbone._swish(backbone._bn1(backbone._conv_head(current_feat)))
                 temp_features.append(x)
                 features = temp_features # Use fallback if endpoints failed
        except AttributeError:
             warnings.warn(f"{backbone_type} might not have `extract_endpoints`. Falling back.")
             # Fallback as above
             x = backbone._swish(backbone._bn0(backbone._conv_stem(x)))
             reduction_indices = [i for i, block in enumerate(backbone._blocks) if block._depthwise_conv.stride == [2, 2]]
             reduction_indices = [0] + reduction_indices + [len(backbone._blocks)-1]
             current_feat = x
             temp_features = []
             for i, block in enumerate(backbone._blocks):
                 current_feat = block(current_feat)
                 if i in reduction_indices:
                     temp_features.append(current_feat)
             x = backbone._swish(backbone._bn1(backbone._conv_head(current_feat)))
             temp_features.append(x)
             features = temp_features


    elif backbone_type.startswith('inception'):
        # Inception_v3 specific feature extraction
        warnings.warn(f"Skip connection feature extraction for {backbone_type} is basic and may need refinement.")
        f = []
        # This requires tracing through the Inception V3 architecture carefully
        # Layer names correspond to torchvision's Inception V3 implementation
        x = backbone.Conv2d_1a_3x3(x); f.append(x) # Example stage 1
        x = backbone.Conv2d_2a_3x3(x)
        x = backbone.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2); f.append(x) # Example stage 2
        x = backbone.Conv2d_3b_1x1(x)
        x = backbone.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2); f.append(x) # Example stage 3
        x = backbone.Mixed_5b(x)
        x = backbone.Mixed_5c(x)
        x = backbone.Mixed_5d(x); f.append(x) # Example stage 4
        x = backbone.Mixed_6a(x)
        x = backbone.Mixed_6b(x)
        x = backbone.Mixed_6c(x)
        x = backbone.Mixed_6d(x)
        x = backbone.Mixed_6e(x)
        # AuxLogits are ignored if aux_logits=False in constructor
        x = backbone.Mixed_7a(x)
        x = backbone.Mixed_7b(x)
        x = backbone.Mixed_7c(x); f.append(x) # Example stage 5 (final encoder output)
        features = f # Return features in order [shallow ... deep]

    elif backbone_type.startswith('clip:'):
         warnings.warn(f"Skip connection feature extraction for CLIP backbone {backbone_type} is not standard UNet practice and likely suboptimal.")
         # CLIP ViT doesn't naturally produce hierarchical features for skips.
         # This extracts the sequence of outputs from the transformer blocks if possible.
         if hasattr(backbone, 'transformer'):
             # Process input projection
             x = backbone.conv1(x)  # spatial -> sequence + class token
             x = x.reshape(x.shape[0], x.shape[1], -1)  # NCHW -> NLC
             x = x.permute(0, 2, 1)  # NLC -> NCL
             x = torch.cat([backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # N(L+1)C
             x = x + backbone.positional_embedding.to(x.dtype)
             x = backbone.ln_pre(x)
             
             # Pass through transformer layers and collect outputs
             intermediate_outputs = []
             current_input = x.permute(1, 0, 2)  # NLC -> LNC
             for i, resblock in enumerate(backbone.transformer.resblocks):
                  current_input = resblock(current_input)
                  # Collect output after certain blocks (e.g., every few layers)
                  # This is arbitrary and needs tuning
                  if i % 3 == 0 or i == len(backbone.transformer.resblocks) - 1: 
                      intermediate_outputs.append(current_input.permute(1, 0, 2)) # LNC -> NLC
             
             # Use the sequence of intermediate outputs as features (might need pooling/reshaping)
             # The spatial structure is lost here, making direct concat difficult
             features = intermediate_outputs # List of NLC tensors
             warnings.warn("CLIP ViT features are sequential (NLC), not spatial (NCHW). Decoder needs adaptation.")
         else: # CLIP ResNet variant (handled by resnet block above if detected correctly)
              warnings.warn(f"Could not identify transformer structure in CLIP backbone {backbone_type}.")
              features = [backbone(x)] # Fallback to final output only


    # Fallback: If no specific handling or errors occurred
    if not features:
        warnings.warn(f"Skip connection feature extraction for {backbone_type} failed or not implemented. Using final backbone output only.")
        # For CNNs, run the modified backbone (no fc layer)
        if not backbone_type.startswith('clip:'): # Avoid running CLIP visual again if fallback
             features = [backbone(x)]
        else: # For CLIP, need to ensure it was run once
             if 'final_features' not in locals(): # Check if CLIP already produced output
                  final_features = backbone.encode_image(x) if hasattr(backbone, 'encode_image') else backbone(x)
             features = [final_features]


    # Ensure features are returned in order: shallowest to deepest for decoder
    # Based on channel dimensions (heuristic)
    if len(features) > 1 and features[0].shape[1] < features[-1].shape[1]:
         pass # Already shallow -> deep
    elif len(features) > 1:
         features = features[::-1] # Reverse if deep -> shallow

    return features


# Simple convolutional block for decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_skip=True, dropout_rate=0.1):
        super().__init__()
        self.use_skip = use_skip
        # Upsampling + Convolution
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Determine channels for conv block input
        conv_in_channels = out_channels + skip_channels if use_skip else out_channels
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate), # Add dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features=None):
        x = self.upconv(x)
        
        if self.use_skip:
            if skip_features is None:
                 raise ValueError("DecoderBlock requires skip_features if use_skip is True")
                 
            # Spatial dimension check/adjustment
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip_features], dim=1)
            
        x = self.conv_block(x)
        return x

class UNetModel(nn.Module):
    """Flexible UNet segmentation model with various backbones."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone, self.backbone_feature_channels = self._create_backbone(config)
        self._modify_input_layer(config, self.backbone)

        if config.freeze_backbone:
            print(f"Freezing backbone: {config.backbone_type}")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create Decoder
        self.decoder_blocks = self._create_decoder(config, self.backbone_feature_channels)

        # Final segmentation head
        final_decoder_channels = config.decoder_channels[-1] if config.decoder_channels else self.backbone_feature_channels[0] # Handle no-decoder case?
        self.segmentation_head = nn.Conv2d(final_decoder_channels, config.num_classes, kernel_size=1)

    def _create_backbone(self, config: ModelConfig) -> tuple[nn.Module, List[int]]:
        """Creates the backbone and returns it along with its feature channel dimensions."""
        backbone_type = config.backbone_type.lower()
        pretrained = config.use_pretrained_backbone
        feature_channels = [] # Stores output channels of layers used for skip connections + final output

        # --- torchvision models ---
        if backbone_type.startswith('resnet'):
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None # Adapt for other variants
            if backbone_type == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = models.resnet18(weights=weights)
                feature_channels = [64, 64, 128, 256, 512] # Channels after stem, layer1-4
            elif backbone_type == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                backbone = models.resnet34(weights=weights)
                feature_channels = [64, 64, 128, 256, 512]
            elif backbone_type == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                backbone = models.resnet50(weights=weights)
                feature_channels = [64, 256, 512, 1024, 2048]
            elif backbone_type == 'resnet101':
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                backbone = models.resnet101(weights=weights)
                feature_channels = [64, 256, 512, 1024, 2048]
            elif backbone_type == 'resnet152':
                weights = models.ResNet152_Weights.DEFAULT if pretrained else None
                backbone = models.resnet152(weights=weights)
                feature_channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone_type}")
            # Remove the final fully connected layer
            backbone.fc = nn.Identity()

        elif backbone_type.startswith('efficientnet'):
            if not _has_efficientnet:
                raise ImportError("efficientnet-pytorch is not installed. Please install it via pip.")
            try:
                # Use 'efficientnet_pytorch' loading
                if pretrained:
                    backbone = EfficientNet.from_pretrained(backbone_type)
                else:
                    backbone = EfficientNet.from_name(backbone_type)
                
                # Remove the final classifier
                backbone._fc = nn.Identity()
                backbone._dropout = nn.Identity()
                backbone._swish = nn.Identity()

                # Determine feature channels (requires inspecting the specific EfficientNet model)
                # Example for B0 (channels after blocks ending stages 1, 2, 3, 4, 5)
                effnet_channels_map = {
                    # Corrected: Exclude the final head channel (1280) for UNet skips
                    'efficientnet-b0': [16, 24, 40, 112, 320], # Stem, S1, S2, S3, S4
                    'efficientnet-b1': [16, 24, 40, 112, 320], # Exclude 1280
                    'efficientnet-b2': [16, 24, 48, 120, 352], # Exclude 1408
                    'efficientnet-b3': [24, 32, 48, 136, 384], # Exclude 1536
                    'efficientnet-b4': [24, 32, 56, 160, 448], # Exclude 1792
                    # Add others if needed, ensuring the last (head) channel is excluded
                }
                if backbone_type in effnet_channels_map:
                    feature_channels = effnet_channels_map[backbone_type]
                else:
                    warnings.warn(f"Feature channels for {backbone_type} not predefined. Attempting heuristic extraction or using default B0 channels.")
                    # Fallback to B0 channels (without head)
                    feature_channels = effnet_channels_map.get('efficientnet-b0', []) 
                    # TODO: Implement a more robust dynamic feature channel extraction if needed

            except Exception as e:
                raise ValueError(f"Could not load EfficientNet {backbone_type}. Error: {e}")

        elif backbone_type.startswith('inception'):
            if backbone_type == 'inception_v3':
                weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
                # Load without aux logits as they are not needed for UNet backbone
                backbone = models.inception_v3(weights=weights, aux_logits=False)
                # Feature channels for Inception V3 (example: after stem, mixed_5d, mixed_6e, mixed_7c)
                # Need to verify these stages correspond to desired UNet skip levels
                feature_channels = [192, 288, 768, 2048] # Example values, verify!
                backbone.fc = nn.Identity() # Remove classification head
            else:
                raise ValueError(f"Unsupported Inception variant: {backbone_type}")

        # --- CLIP model ---
        elif backbone_type.startswith('clip:'):
            if not _has_clip:
                raise ImportError("CLIP library is not installed. Please install `ftfy regex tqdm`, then `pip install git+https://github.com/openai/CLIP.git`")
            clip_model_name = backbone_type.split(':')[1]
            try:
                print(f"Loading CLIP model: {clip_model_name}")
                # Load CLIP model on CPU first to avoid device issues during init
                clip_model, _ = clip.load(clip_model_name, device='cpu', jit=False) # jit=False sometimes helps compatibility
                backbone = clip_model.visual
                print(f"Using CLIP visual backbone of type: {type(backbone)}")

                # Determine feature channels based on CLIP visual encoder type
                if isinstance(backbone, models.resnet.ResNet):
                    print("CLIP backbone is ResNet-based.")
                    # Use ResNet feature channel logic
                    base_channels = 64 # Typically 64 for ResNet stem
                    layer_mults = []
                    if hasattr(backbone.layer1[0], 'expansion'): # Bottleneck blocks
                         exp = backbone.layer1[0].expansion
                         layer_mults = [exp, exp*2, exp*4, exp*8]
                    else: # Basic blocks
                         layer_mults = [1, 2, 4, 8]

                    feature_channels = [base_channels] + [base_channels * m for m in layer_mults]
                    # Remove the attention pooling layer specific to CLIP's ResNet
                    backbone.attnpool = nn.Identity()
                    print(f"Determined ResNet-CLIP feature channels: {feature_channels}")

                elif hasattr(backbone, 'transformer'): # ViT-based CLIP model
                    print("CLIP backbone is ViT-based.")
                    # ViT features are sequence-based, not naturally hierarchical for UNet skips
                    # Option 1: Use only the final feature map (output_dim)
                    # Option 2: Try to extract features from intermediate transformer blocks (complex)
                    # For simplicity, we'll primarily rely on the final output dimension
                    final_dim = backbone.output_dim
                    # We might fabricate intermediate dimensions for the decoder, but it's artificial
                    feature_channels = [final_dim // 4, final_dim // 2, final_dim] # Fabricated example
                    # feature_channels = [final_dim] # Simplest approach
                    warnings.warn("Using CLIP ViT backbone. Feature channels for UNet skips are experimental/approximated.")
                    # Need to handle positional embeddings and class token carefully in forward pass / feature extraction
                else:
                     raise ValueError(f"Unsupported CLIP visual backbone architecture: {type(backbone)}")
            except Exception as e:
                raise ValueError(f"Could not load CLIP model {clip_model_name}. Is it installed and available? Error: {e}")


        # --- SAM model (Placeholder) ---
        elif backbone_type == 'sam2':
            if not _has_sam:
                raise ImportError("Segment Anything Model (SAM) library not installed or integrated. See https://github.com/facebookresearch/segment-anything")
            if config.sam_checkpoint_path is None:
                raise ValueError("sam_checkpoint_path must be provided in config for sam2 backbone.")
            # Example using segment-anything library (replace with actual API usage)
            # try:
            #     from segment_anything import sam_model_registry
            #     # Determine model type based on checkpoint name or config (e.g., 'vit_h', 'vit_l', 'vit_b')
            #     sam_model_type = "default" # Or determine dynamically
            #     sam = sam_model_registry[sam_model_type](checkpoint=config.sam_checkpoint_path)
            #     backbone = sam.image_encoder
            #     # Determine SAM feature channels (depends on ViT variant and internal structure)
            #     # Example for ViT-H: Might involve output after specific transformer blocks
            #     feature_channels = [256, 512, 1024, 1280] # Example, needs verification for SAM's encoder
            #     warnings.warn("SAM feature channels are placeholders and need verification.")
            # except Exception as e:
            #     raise ValueError(f"Failed to load SAM model: {e}")
            raise NotImplementedError("SAM2 backbone integration is not fully implemented.")

        # --- TIMM models (Optional Extension) ---
        elif _has_timm and backbone_type in timm.list_models(pretrained=pretrained):
             try:
                 print(f"Loading TIMM model: {backbone_type}")
                 # Load backbone with features_only=True to get intermediate outputs easily
                 backbone = timm.create_model(
                     backbone_type,
                     pretrained=pretrained,
                     features_only=True,
                     in_chans=config.in_channels # Pass input channels directly to timm
                 )
                 # Get feature channels from timm's feature info
                 feature_channels = backbone.feature_info.channels()
                 print(f"TIMM feature channels: {feature_channels}")
                 # No need to modify input layer if timm handles in_chans
                 # No need to remove fc layer if features_only=True
             except Exception as e:
                 raise ValueError(f"Could not load TIMM model {backbone_type}. Error: {e}")

        else:
            supported = "torchvision (resnet*, efficientnet*, inception_v3), clip:<model_name>, sam2"
            if _has_timm: supported += ", timm models"
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported families: {supported}")

        # Final check for feature channels
        if not feature_channels:
             raise RuntimeError(f"Could not determine feature channels for backbone {backbone_type}.")

        print(f"Created backbone: {backbone_type} with feature channels: {feature_channels}")
        return backbone, feature_channels


    def _modify_input_layer(self, config: ModelConfig, backbone: nn.Module):
        """Modify the input layer of the backbone if necessary."""
        # If TIMM handled in_chans, skip modification
        if _has_timm and hasattr(timm.models, '_features') and isinstance(backbone, timm.models._features.FeatureListNet):
             print("TIMM model created with specified input channels. Skipping input layer modification.")
             return

        if config.in_channels == 3: # Default for many pretrained models
            print("Input channels is 3. Assuming standard backbone input layer.")
            return

        target_layer = None
        layer_name = ""
        parent_module = backbone
        module_name_prefix = "" # For nested modules like clip.visual

        # --- Find the target Conv2d layer ---
        # Check for CLIP visual encoder structure first
        if config.backbone_type.startswith('clip:') and hasattr(backbone, 'conv1'):
             target_layer = backbone.conv1
             layer_name = 'conv1'
             parent_module = backbone # Already points to visual encoder
             print(f"Found target layer in CLIP visual: {layer_name}")
        elif isinstance(backbone, models.Inception3):
             target_layer = backbone.Conv2d_1a_3x3.conv
             layer_name = 'Conv2d_1a_3x3.conv'
             parent_module = backbone # Direct attribute
             print(f"Found target layer in InceptionV3: {layer_name}")
        else:
            # Common first conv layer names/paths
            potential_paths = ['conv1', 'features.0.0', '_conv_stem', 'stem.0', 'conv_proj']
            for path in potential_paths:
                try:
                    module = backbone
                    parts = path.split('.')
                    for i, part in enumerate(parts):
                        if i == len(parts) - 1: # Last part is the layer name
                             if hasattr(module, part):
                                  layer = getattr(module, part)
                                  if isinstance(layer, nn.Conv2d):
                                       target_layer = layer
                                       layer_name = part
                                       parent_module = module
                                       print(f"Found target layer at path: {path}")
                                       break
                             # Handle potential sequence access like features[0][0] - simplified
                             elif '[' in part and part.endswith(']'):
                                 base, index = part[:-1].split('[')
                                 idx = int(index)
                                 seq_module = getattr(module, base)
                                 if isinstance(seq_module[idx], nn.Conv2d):
                                     target_layer = seq_module[idx]
                                     layer_name = part # Keep original notation for replacement? Or just index?
                                     parent_module = seq_module # Parent is the sequence
                                     print(f"Found target layer in sequence: {path}")
                                     break
                        else: # Traverse module path
                             module = getattr(module, part)
                    if target_layer: break # Found it
                except (AttributeError, IndexError, TypeError):
                    continue # Path doesn't exist or isn't valid

        # --- Modify the found layer ---
        if target_layer and isinstance(target_layer, nn.Conv2d):
            original_channels = target_layer.in_channels
            if original_channels != config.in_channels:
                print(f"Modifying input layer '{layer_name}' for {config.backbone_type} "
                      f"to accept {config.in_channels} channels (originally {original_channels}).")
                
                new_conv = nn.Conv2d(
                    in_channels=config.in_channels,
                    out_channels=target_layer.out_channels,
                    kernel_size=target_layer.kernel_size,
                    stride=target_layer.stride,
                    padding=target_layer.padding,
                    dilation=target_layer.dilation,
                    groups=target_layer.groups,
                    bias=(target_layer.bias is not None)
                )

                # Weight initialization heuristic
                with torch.no_grad():
                    if original_channels == 3:
                        # Average RGB weights for extra channels
                        avg_weights = target_layer.weight.data.mean(dim=1, keepdim=True)
                        new_conv.weight.data[:, :3, :, :] = target_layer.weight.data
                        for i in range(3, config.in_channels):
                            new_conv.weight.data[:, i:i+1, :, :] = avg_weights
                    else: # Other cases: copy what fits, rest uses default init
                        copy_channels = min(original_channels, config.in_channels)
                        new_conv.weight.data[:, :copy_channels, :, :] = target_layer.weight.data[:, :copy_channels, :, :]

                    if target_layer.bias is not None:
                        new_conv.bias.data = target_layer.bias.data

                # Replace the layer in the parent module
                try:
                     # Handle direct attribute vs. sequence index
                     if '[' in layer_name and layer_name.endswith(']'):
                         base, index = layer_name[:-1].split('[')
                         idx = int(index)
                         # parent_module should already be the sequence
                         parent_module[idx] = new_conv
                     else:
                         setattr(parent_module, layer_name, new_conv)
                     print(f"Successfully replaced layer '{layer_name}'.")
                except Exception as e:
                     warnings.warn(f"Failed to automatically replace input layer '{layer_name}'. "
                                   f"Manual adjustment might be needed. Error: {e}")
            else:
                 print(f"Input layer '{layer_name}' already has {config.in_channels} channels.")
        else:
            warnings.warn(f"Could not find a standard input Conv2d layer to modify for "
                          f"{config.backbone_type} with {config.in_channels} channels. "
                          f"Model might fail if backbone expects 3 channels.")


    def _create_decoder(self, config: ModelConfig, backbone_feature_channels: List[int]):
        """Creates the UNet decoder blocks."""
        decoder_blocks = nn.ModuleList()

        if not backbone_feature_channels:
             raise ValueError("Backbone feature channels list is empty. Cannot create decoder.")

        # Feature channels from encoder, ordered from deepest to shallowest stage output
        encoder_channels = backbone_feature_channels[::-1]

        num_encoder_stages = len(encoder_channels)
        num_decoder_blocks = len(config.decoder_channels)

        # Typically, num_decoder_blocks == num_encoder_stages - 1
        if num_decoder_blocks != num_encoder_stages - 1 and config.use_skip_connections:
             warnings.warn(
                 f"Number of decoder channel specifications ({num_decoder_blocks}) "
                 f"does not match the expected number of decoder blocks ({num_encoder_stages - 1}) "
                 f"for backbone {config.backbone_type} with {num_encoder_stages} feature levels. "
                 f"Decoder config: {config.decoder_channels}, Encoder channels: {backbone_feature_channels}. "
                 "Adjusting decoder_channels list length."
             )
             # Simple fix: truncate or pad decoder_channels list
             target_len = num_encoder_stages - 1
             if num_decoder_blocks > target_len:
                 config.decoder_channels = config.decoder_channels[:target_len]
             else:
                 config.decoder_channels += [config.decoder_channels[-1]] * (target_len - num_decoder_blocks)
             num_decoder_blocks = len(config.decoder_channels) # Update length

        in_ch = encoder_channels[0] # Channel dim of the deepest encoder feature map

        for i in range(num_decoder_blocks):
            skip_ch = encoder_channels[i+1] if config.use_skip_connections else 0
            out_ch = config.decoder_channels[i]

            decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch,
                             use_skip=config.use_skip_connections,
                             dropout_rate=config.dropout_rate)
            )
            in_ch = out_ch # Input to the next decoder block is the output of the current one

        return decoder_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder, decoder with skip connections."""

        # --- Encoder ---
        # Use the refined get_intermediate_features helper
        encoder_features = get_intermediate_features(self.config.backbone_type, self.backbone, x)

        # Validate number of features extracted vs expected
        num_expected_features = len(self.backbone_feature_channels)
        if len(encoder_features) != num_expected_features:
             warnings.warn(
                 f"Expected {num_expected_features} feature maps from backbone {self.config.backbone_type}, "
                 f"but got {len(encoder_features)}. Skip connections might mismatch."
                 f"Feature shapes: {[f.shape for f in encoder_features]}"
             )
             # Attempt to recover if possible, e.g., by selecting subset or raising error
             if len(encoder_features) < num_expected_features:
                  raise RuntimeError("Insufficient feature maps extracted from backbone for decoder.")
             # If too many, select the ones most likely corresponding to channel list? Risky.
             # For now, proceed with warning.

        # --- Decoder ---
        # Start with the deepest feature map
        decoder_output = encoder_features[-1]

        # Iterate through decoder blocks and skip connections (shallowest feature is not used as skip)
        num_skips = len(encoder_features) - 1
        for i in range(len(self.decoder_blocks)):
            if self.config.use_skip_connections and i < num_skips:
                # Skip features are from shallower encoder stages
                skip_index = num_skips - 1 - i # e.g., for block 0, skip is index -2; for block 1, skip is index -3
                skip_features = encoder_features[skip_index]
                # Handle potential sequential features from ViT backbones (requires adaptation)
                if skip_features.ndim == 3: # NLC format from ViT
                    warnings.warn("Decoder received sequential features (NLC) from ViT backbone. Needs spatial adaptation (placeholder).")
                    # Placeholder: Reshape/pool NLC to NCHW - This is non-trivial and model-specific
                    # Might involve reshaping based on patch size, or using a learned adapter.
                    # Simple avg pooling as a fallback:
                    num_tokens = skip_features.shape[1]
                    embed_dim = skip_features.shape[2]
                    # Try to guess spatial dimensions (e.g., sqrt(num_tokens-1) if class token exists)
                    h = w = int((num_tokens-1)**0.5) if (num_tokens-1)**0.5 == int((num_tokens-1)**0.5) else None
                    if h and w:
                         skip_features = skip_features[:, 1:, :].permute(0, 2, 1).reshape(skip_features.shape[0], embed_dim, h, w)
                    else: # Fallback: adaptive avg pool (loses spatial info)
                         skip_features = F.adaptive_avg_pool1d(skip_features.permute(0,2,1), 1).unsqueeze(-1) # N C 1 1
            else:
                skip_features = None

            # Pass through decoder block
            decoder_output = self.decoder_blocks[i](decoder_output, skip_features)


        # --- Segmentation Head ---
        logits = self.segmentation_head(decoder_output)

        # --- Final Upsampling (Optional) ---
        # Upsample logits to match input size if needed (UNet often outputs smaller resolution)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits


# --- EfficientNet Helper (Optional, if needed and reliable) ---
# Monkey-patching can be fragile; consider external helper functions instead.
# The `extract_endpoints` method in efficientnet_pytorch might be more robust if available.
# if _has_efficientnet and EfficientNet is not None:
#     def get_efficientnet_feature_channels_helper(model: EfficientNet) -> List[int]:
#         # Implementation to reliably get channels after each reduction stage
#         # ... (requires detailed knowledge of efficientnet_pytorch internal structure)
#         pass

#     # Example of patching if the helper is defined:
#     # try:
#     #     if not hasattr(EfficientNet, 'get_feature_channels'):
#     #          EfficientNet.get_feature_channels = get_efficientnet_feature_channels_helper
#     # except Exception as e:
#     #     warnings.warn(f"Could not monkey-patch get_feature_channels onto EfficientNet: {e}")

 