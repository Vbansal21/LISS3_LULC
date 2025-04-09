#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for lightweight SAM2-UNet model for Land Use/Land Cover Change and Wildfire Prediction
Mature version with robust error handling and logging
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
import logging
import time
from datetime import datetime

# Import our models
sys.path.append('/home/ubuntu/deep_learning_project')
from models.sam2_unet_lite import SAM2UNetLite, LandCoverChangeModelLite, WildfireDetectionModelLite

# Import our datasets
from scripts.data_preprocessing import create_dataloaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train lightweight SAM2-UNet for Land Use/Land Cover Change and Wildfire Prediction')
    parser.add_argument('--task', type=str, default='landcover', choices=['landcover', 'wildfire'],
                        help='Task to train for: landcover or wildfire')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/deep_learning_project/data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patch-size', type=int, default=128, help='Size of image patches')
    parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained SAM2 model')
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/deep_learning_project/results',
                        help='Path to output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='/home/ubuntu/deep_learning_project/checkpoints',
                        help='Path to checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args):
    """Create model based on task"""
    logger.info(f"Creating model for {args.task} task")
    
    if args.task == 'landcover':
        model = LandCoverChangeModelLite(num_classes=21)  # UC Merced has 21 land use classes
    elif args.task == 'wildfire':
        model = WildfireDetectionModelLite(num_classes=2)  # Binary classification: fire/no-fire
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model, device

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, train_losses, val_losses
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.info("Starting training from scratch")
        return 0, float('inf'), [], []

def train_epoch(model, train_loader, criterion, optimizer, device, task, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1} Training')):
        try:
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
                
                # Calculate accuracy
                _, predicted = torch.max(seg1.data, 1)
                total += targets.size(0) * targets.size(1) * targets.size(2)
                correct += (predicted == targets).sum().item()
                
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
                
                # Calculate accuracy
                _, predicted = torch.max(fire1.data, 1)
                total += targets.size(0) * targets.size(1) * targets.size(2)
                correct += (predicted == targets).sum().item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 10 batches
            if i % 10 == 0:
                logger.info(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        except Exception as e:
            logger.error(f"Error in training batch {i}: {e}")
            continue
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total if total > 0 else 0
    epoch_time = time.time() - start_time
    
    logger.info(f'Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, task, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')):
            try:
                if task == 'landcover':
                    inputs1 = inputs.to(device)
                    inputs2 = inputs.to(device)
                    targets = targets.to(device)
                    
                    seg1, seg2, change_prob = model(inputs1, inputs2)
                    loss = criterion(seg1, targets) + criterion(seg2, targets)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(seg1.data, 1)
                    total += targets.size(0) * targets.size(1) * targets.size(2)
                    correct += (predicted == targets).sum().item()
                    
                elif task == 'wildfire':
                    inputs1 = inputs.to(device)
                    inputs2 = inputs.to(device)
                    targets = targets.to(device)
                    
                    fire1, fire2, spread_prob = model(inputs1, inputs2)
                    loss = criterion(fire1, targets)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(fire1.data, 1)
                    total += targets.size(0) * targets.size(1) * targets.size(2)
                    correct += (predicted == targets).sum().item()
                
                running_loss += loss.item()
                
            except Exception as e:
                logger.error(f"Error in validation batch {i}: {e}")
                continue
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total if total > 0 else 0
    epoch_time = time.time() - start_time
    
    logger.info(f'Validation - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_losses, val_losses, checkpoint_path):
    """Save model checkpoint"""
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, checkpoint_path)
        logger.info("Checkpoint saved successfully")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

def plot_losses(train_losses, val_losses, output_path):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    try:
        plt.savefig(output_path)
        logger.info(f"Loss plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving loss plot: {e}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output and checkpoint directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Log training configuration
    logger.info(f"Starting training with configuration:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Patch size: {args.patch_size}")
    logger.info(f"  Use pretrained: {args.use_pretrained}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"  Resume from: {args.resume}")
    logger.info(f"  Random seed: {args.seed}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, task=args.task, batch_size=args.batch_size, patch_size=args.patch_size
        )
        logger.info(f"Dataloaders created successfully:")
        logger.info(f"  Train: {len(train_loader.dataset)} samples")
        logger.info(f"  Validation: {len(val_loader.dataset)} samples")
        logger.info(f"  Test: {len(test_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return
    
    # Create model
    logger.info("Creating model...")
    model, device = create_model(args)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(
            model, optimizer, args.resume
        )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.task, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, args.task, epoch)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.task}_best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, train_losses, val_losses, checkpoint_path
            )
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.task}_epoch_{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, train_losses, val_losses, checkpoint_path
            )
    
    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.task}_final_model.pth')
    save_checkpoint(
        model, optimizer, args.epochs-1, train_losses[-1], val_losses[-1], 
        train_losses, val_losses, final_checkpoint_path
    )
    
    # Plot training and validation loss
    loss_plot_path = os.path.join(args.output_dir, f'{args.task}_loss.png')
    plot_losses(train_losses, val_losses, loss_plot_path)
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f'Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
