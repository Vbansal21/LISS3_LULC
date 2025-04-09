# U-Net Architecture with Foundation Models for Land Use/Land Cover Change and Wildfire Prediction

## Research Summary

This document summarizes research on integrating U-Net architecture with foundation models (SAM/CLIP) for land use/land cover change and wildfire prediction using Landsat 8-9 imagery.

### 1. U-Net Architecture

U-Net is a convolutional neural network architecture designed for image segmentation tasks. Key characteristics:

- Encoder-decoder structure with a contracting path (encoder) and an expansive path (decoder)
- Skip connections between corresponding encoder and decoder layers
- Ability to work with limited training data through data augmentation
- Originally designed for biomedical image segmentation but widely adapted for various domains

The U-Net architecture is particularly suitable for our task because:
- It can effectively segment different land cover types
- It works well with satellite imagery
- It can be trained to detect changes over time
- It can be adapted to identify wildfire-affected areas

### 2. Segment Anything Model (SAM) Integration

SAM is a foundation model for image segmentation that can be integrated with U-Net:

#### SAM2-UNet Approach
- Uses the Hiera backbone of SAM2 as the encoder for a U-Net architecture
- Maintains the classic U-shaped design for the decoder
- Inserts adapters into the encoder for parameter-efficient fine-tuning
- Demonstrated strong performance across various segmentation tasks

#### Advantages of SAM Integration:
- Pre-trained on large datasets, providing strong feature extraction capabilities
- Zero-shot segmentation abilities that can be fine-tuned for specific domains
- Robust performance on complex boundaries and shapes

#### Limitations:
- Some studies show that standard U-Net can outperform SAM in specific domains
- May require significant adaptation for remote sensing applications

### 3. CLIP Model Integration

CLIP (Contrastive Language-Image Pre-training) can also be integrated with U-Net:

#### Potential Approaches:
- Using CLIP's image encoder as the encoder portion of U-Net
- Leveraging CLIP's multimodal capabilities to incorporate textual descriptions
- Adapting CLIP's contrastive learning approach for change detection

#### Advantages of CLIP Integration:
- Strong visual representation learning
- Ability to incorporate textual descriptions of land cover types
- Zero-shot classification capabilities

### 4. Temporal Information Integration

For land use/land cover change and wildfire prediction, temporal information is crucial:

#### Temporal Prompt Guided U-Net (TP-UNet):
- Uses temporal prompts to guide the segmentation model
- Incorporates cross-attention mechanisms to combine temporal information with image features
- Leverages contrastive learning for semantic alignment

This approach could be particularly valuable for our task as it would allow the model to detect changes over time, which is essential for both land cover change analysis and wildfire prediction.

## Recommended Approach

Based on the research, the most promising approach for our task is:

1. **SAM2-UNet with Temporal Integration**:
   - Use SAM2 as the encoder backbone for strong feature extraction
   - Implement a U-Net decoder with skip connections
   - Add temporal integration mechanisms to capture changes over time
   - Fine-tune on our specific datasets (UC Merced, ESA WorldCover, and Landsat)

2. **Implementation Considerations**:
   - Parameter-efficient fine-tuning using adapters
   - Custom loss functions for both segmentation and change detection
   - Data augmentation strategies specific to satellite imagery
   - Multi-scale feature fusion for handling different resolution inputs

This approach combines the strengths of foundation models with the proven effectiveness of U-Net for segmentation tasks, while also addressing the temporal aspects necessary for change detection and prediction.
