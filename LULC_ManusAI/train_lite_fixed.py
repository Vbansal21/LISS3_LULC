#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for lightweight SAM2-UNet model for Land Use/Land Cover Change and Wildfire Prediction
Fixed version to address dimension mismatch
"""

import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our models
from sam2_unet_lite import SAM2UNetLite, LandCoverChangeModelLite, WildfireDetectionModelLite

# Import our datasets
# sys.path.append('/home/ubuntu/deep_learning_project')
from data_preprocessing import UCMercedDataset, ESAWorldCoverDataset, LandsatDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train lightweight SAM2-UNet for Land Use/Land Cover Change and Wildfire Prediction')
    parser.add_argument('--task', type=str, default='landcover', choices=['landcover', 'wildfire'],
                        help='Task to train for: landcover or wildfire')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/deep_learning_project/data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained SAM2 model')
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/deep_learning_project/results',
                        help='Path to output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='/home/ubuntu/deep_learning_project/checkpoints',
                        help='Path to checkpoint directory')
    return parser.parse_args()

class ResizeTarget(nn.Module):
    """Resize target tensor to match model output dimensions"""
    def __init__(self, size=(128, 128)):
        super(ResizeTarget, self).__init__()
        self.size = size
    
    def forward(self, x):
        return nn.functional.interpolate(x.unsqueeze(1).float(), size=self.size, mode='nearest').squeeze(1).long()

def create_dataloaders(args):
    """Create dataloaders for training and validation"""
    if args.task == 'landcover':
        # For land cover change prediction, we use UC Merced dataset
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((128, 128), antialias=True)  # Resize to smaller dimensions to save memory
        ])
        
        dataset = UCMercedDataset(
            root_dir=os.path.join(args.data_dir, 'UCMerced_LandUse/UCMerced_LandUse/Images'),
            transform=transform
        )
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
    elif args.task == 'wildfire':
        # For wildfire prediction, we use Landsat dataset
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),  # 4 channels: RGB + NBR
            transforms.Resize((128, 128), antialias=True)  # Resize to smaller dimensions to save memory
        ])
        
        # This is a placeholder - in a real implementation, we would use the actual Landsat dataset
        dataset = LandsatDataset(
            tif_path=os.path.join(args.data_dir, 'Landsat/LC09_NBR_20250114_182831_041036.tif'),
            transform=transform
        )
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def create_model(args):
    """Create model based on task"""
    if args.task == 'landcover':
        model = LandCoverChangeModelLite(num_classes=21)  # UC Merced has 21 land use classes
    elif args.task == 'wildfire':
        model = WildfireDetectionModelLite(num_classes=2)  # Binary classification: fire/no-fire
    
    # Use CPU for training (as GPU is not available)
    device = torch.device('cpu')
    model = model.to(device)
    
    return model, device

def train_epoch(model, train_loader, criterion, optimizer, device, task, target_resize):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training')):
        # In a real implementation, we would handle the data properly
        # For now, we'll use a placeholder approach
        
        # Resize targets to match model output dimensions
        targets = target_resize(targets)
        
        if task == 'landcover':
            # For land cover change, we need two time points
            # For placeholder purposes, we'll just duplicate the input
            inputs1 = inputs.to(device)
            inputs2 = inputs.to(device)  # In reality, this would be a different time point
            targets = targets.to(device)
            
            # Forward pass
            seg1, seg2, change_prob = model(inputs1, inputs2)
            
            # Calculate loss (placeholder - would be more complex in reality)
            loss = criterion(seg1, targets) + criterion(seg2, targets)
            
        elif task == 'wildfire':
            # For wildfire detection, we also need two time points for spread prediction
            # For placeholder purposes, we'll just duplicate the input
            inputs1 = inputs.to(device)
            inputs2 = inputs.to(device)  # In reality, this would be a different time point
            targets = targets.to(device)
            
            # Forward pass
            fire1, fire2, spread_prob = model(inputs1, inputs2)
            
            # Calculate loss (placeholder - would be more complex in reality)
            loss = criterion(fire1, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 10 batches
        if i % 10 == 0:
            print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device, task, target_resize):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(val_loader, desc='Validation')):
            # Resize targets to match model output dimensions
            targets = target_resize(targets)
            
            # Similar placeholder approach as in train_epoch
            if task == 'landcover':
                inputs1 = inputs.to(device)
                inputs2 = inputs.to(device)
                targets = targets.to(device)
                
                seg1, seg2, change_prob = model(inputs1, inputs2)
                loss = criterion(seg1, targets) + criterion(seg2, targets)
                
            elif task == 'wildfire':
                inputs1 = inputs.to(device)
                inputs2 = inputs.to(device)
                targets = targets.to(device)
                
                fire1, fire2, spread_prob = model(inputs1, inputs2)
                loss = criterion(fire1, targets)
            
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def main():
    # Create output and checkpoint directories
    os.makedirs('/home/ubuntu/deep_learning_project/results', exist_ok=True)
    os.makedirs('/home/ubuntu/deep_learning_project/checkpoints', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='landcover', choices=['landcover', 'wildfire'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/deep_learning_project/results')
    parser.add_argument('--checkpoint-dir', type=str, default='/home/ubuntu/deep_learning_project/checkpoints')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/deep_learning_project/data')
    parser.add_argument('--use-pretrained', action='store_true')
    args = parser.parse_args()
    
    print(f"Training lightweight SAM2-UNet for {args.task} task")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.lr}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(args)
    
    # Create model
    print("Creating model...")
    model, device = create_model(args)
    
    # Create target resize module
    target_resize = ResizeTarget(size=(128, 128))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.task, target_resize)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, args.task, target_resize)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.task}_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.task.capitalize()} Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'{args.task}_loss.png'))
    
    print('Training completed!')

if __name__ == '__main__':
    main()
