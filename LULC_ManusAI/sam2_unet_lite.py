#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM2-UNet Model Implementation for Land Use/Land Cover Change and Wildfire Prediction
Lightweight version to address memory constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d

class ConvBlock(nn.Module):
    """Basic convolutional block with double convolution"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    """Encoder module - lightweight version"""
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        # Reduced feature dimensions to save memory
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Store intermediate outputs for skip connections
        features = []
        
        x1 = self.enc1(x)
        features.append(x1)
        x = self.pool(x1)
        
        x2 = self.enc2(x)
        features.append(x2)
        x = self.pool(x2)
        
        x3 = self.enc3(x)
        features.append(x3)
        x = self.pool(x3)
        
        x4 = self.enc4(x)
        features.append(x4)
        
        return features

class Adapter(nn.Module):
    """Adapter module for parameter-efficient fine-tuning"""
    def __init__(self, in_channels, reduction_factor=4):
        super(Adapter, self).__init__()
        self.down_project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_factor, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up_project = nn.Conv2d(in_channels // reduction_factor, in_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.up_project(x)
        return residual + x

class TemporalAttention(nn.Module):
    """Simplified temporal attention module"""
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.conv = Conv2d(channels * 2, channels, kernel_size=1)
        
    def forward(self, x, temporal_prompt):
        # Simple concatenation and convolution instead of full attention
        combined = torch.cat([x, temporal_prompt], dim=1)
        out = self.conv(combined)
        return out

class Decoder(nn.Module):
    """U-Net decoder with skip connections - lightweight version"""
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        
        # Upsampling path
        self.up_conv1 = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Convolutional blocks after concatenation with skip connections
        self.dec1 = ConvBlock(256, 128)
        self.dec2 = ConvBlock(128, 64)
        self.dec3 = ConvBlock(64, 32)
        
        # Temporal attention modules
        self.temp_attn1 = TemporalAttention(128)
        self.temp_attn2 = TemporalAttention(64)
        self.temp_attn3 = TemporalAttention(32)
        
        # Final output layer
        self.final_conv = Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, features, temporal_features=None):
        # If no temporal features are provided, use the same features
        if temporal_features is None:
            temporal_features = features
        
        # Decoder with skip connections and temporal attention
        x = self.up_conv1(features[3])
        x = torch.cat([x, features[2]], dim=1)
        x = self.dec1(x)
        x = self.temp_attn1(x, temporal_features[2])
        
        x = self.up_conv2(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.dec2(x)
        x = self.temp_attn2(x, temporal_features[1])
        
        x = self.up_conv3(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.dec3(x)
        x = self.temp_attn3(x, temporal_features[0])
        
        # Final output
        output = self.final_conv(x)
        
        return output

class SAM2UNetLite(nn.Module):
    """
    Lightweight SAM2-UNet model for land use/land cover change and wildfire prediction
    """
    def __init__(self, in_channels=3, num_classes=2, use_pretrained_sam2=False):
        super(SAM2UNetLite, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initialize encoder
        self.encoder = Encoder(in_channels)
        
        # Add adapters to each encoder layer if using pretrained model
        if use_pretrained_sam2:
            self.adapters = nn.ModuleList([
                Adapter(32),
                Adapter(64),
                Adapter(128),
                Adapter(256)
            ])
        else:
            self.adapters = None
        
        # Initialize decoder
        self.decoder = Decoder(num_classes)
        
    def forward(self, x, prev_x=None):
        # Extract features from current input
        features = self.encoder(x)
        
        # Apply adapters if available
        if self.adapters is not None:
            adapted_features = []
            for i, feature in enumerate(features):
                adapted_features.append(self.adapters[i](feature))
            features = adapted_features
        
        # Extract features from previous time step if available
        temporal_features = None
        if prev_x is not None:
            temporal_features = self.encoder(prev_x)
            
            # Apply adapters to temporal features if available
            if self.adapters is not None:
                adapted_temporal_features = []
                for i, feature in enumerate(temporal_features):
                    adapted_temporal_features.append(self.adapters[i](feature))
                temporal_features = adapted_temporal_features
        
        # Decode features with temporal attention
        output = self.decoder(features, temporal_features)
        
        return output

class LandCoverChangeModelLite(nn.Module):
    """
    Lightweight model for land use/land cover change prediction
    """
    def __init__(self, num_classes=21):  # UC Merced has 21 land use classes
        super(LandCoverChangeModelLite, self).__init__()
        self.sam2_unet = SAM2UNetLite(in_channels=3, num_classes=num_classes)
        
        # Additional layer to predict change probability
        self.change_head = nn.Sequential(
            Conv2d(num_classes * 2, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x1, x2):
        # Get segmentation for both time points
        seg1 = self.sam2_unet(x1)
        seg2 = self.sam2_unet(x2, x1)  # Use x1 as temporal context
        
        # Concatenate segmentations to predict change
        concat_seg = torch.cat([seg1, seg2], dim=1)
        change_prob = self.change_head(concat_seg)
        
        return seg1, seg2, change_prob

class WildfireDetectionModelLite(nn.Module):
    """
    Lightweight model for wildfire detection and prediction
    """
    def __init__(self, num_classes=2):  # Binary classification: fire/no-fire
        super(WildfireDetectionModelLite, self).__init__()
        self.sam2_unet = SAM2UNetLite(in_channels=4, num_classes=num_classes)  # RGB + NBR
        
        # Additional layer to predict fire spread probability
        self.spread_head = nn.Sequential(
            Conv2d(num_classes * 2, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x1, x2=None):
        # If only one time point is provided
        if x2 is None:
            return self.sam2_unet(x1)
        
        # Get fire detection for both time points
        fire1 = self.sam2_unet(x1)
        fire2 = self.sam2_unet(x2, x1)  # Use x1 as temporal context
        
        # Predict fire spread
        concat_fire = torch.cat([fire1, fire2], dim=1)
        spread_prob = self.spread_head(concat_fire)
        
        return fire1, fire2, spread_prob

def test_models():
    """Test the implemented models with random inputs - using smaller batch size and image dimensions"""
    # Test SAM2UNetLite
    model = SAM2UNetLite(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 channels, 128x128 resolution
    prev_x = torch.randn(1, 3, 128, 128)
    output = model(x, prev_x)
    print(f"SAM2UNetLite output shape: {output.shape}")
    
    # Test LandCoverChangeModelLite
    lc_model = LandCoverChangeModelLite(num_classes=21)
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 128, 128)
    seg1, seg2, change_prob = lc_model(x1, x2)
    print(f"Land Cover Change Model output shapes: {seg1.shape}, {seg2.shape}, {change_prob.shape}")
    
    # Test WildfireDetectionModelLite
    wf_model = WildfireDetectionModelLite(num_classes=2)
    x1 = torch.randn(1, 4, 128, 128)  # 4 channels: RGB + NBR
    x2 = torch.randn(1, 4, 128, 128)
    fire1, fire2, spread_prob = wf_model(x1, x2)
    print(f"Wildfire Detection Model output shapes: {fire1.shape}, {fire2.shape}, {spread_prob.shape}")

if __name__ == "__main__":
    test_models()
