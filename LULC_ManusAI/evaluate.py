#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for lightweight SAM2-UNet model for Land Use/Land Cover Change and Wildfire Prediction
"""

import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, jaccard_score
import seaborn as sns
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
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate lightweight SAM2-UNet for Land Use/Land Cover Change and Wildfire Prediction')
    parser.add_argument('--task', type=str, default='landcover', choices=['landcover', 'wildfire'],
                        help='Task to evaluate: landcover or wildfire')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/deep_learning_project/data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--patch-size', type=int, default=128, help='Size of image patches')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/deep_learning_project/results',
                        help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):
    """Load model from checkpoint"""
    logger.info(f"Loading model for {args.task} task from {args.checkpoint}")
    
    try:
        if args.task == 'landcover':
            model = LandCoverChangeModelLite(num_classes=21)  # UC Merced has 21 land use classes
        elif args.task == 'wildfire':
            model = WildfireDetectionModelLite(num_classes=2)  # Binary classification: fire/no-fire
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from epoch {checkpoint['epoch']+1}")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def evaluate(model, test_loader, device, task, output_dir):
    """Evaluate the model"""
    logger.info("Starting evaluation...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc='Evaluation')):
            try:
                if task == 'landcover':
                    inputs1 = inputs.to(device)
                    inputs2 = inputs.to(device)  # In reality, this would be a different time point
                    targets = targets.to(device)
                    
                    # Forward pass
                    seg1, seg2, change_prob = model(inputs1, inputs2)
                    
                    # Get predictions
                    _, preds = torch.max(seg1, 1)
                    
                elif task == 'wildfire':
                    inputs1 = inputs.to(device)
                    inputs2 = inputs.to(device)  # In reality, this would be a different time point
                    targets = targets.to(device)
                    
                    # Forward pass
                    fire1, fire2, spread_prob = model(inputs1, inputs2)
                    
                    # Get predictions
                    _, preds = torch.max(fire1, 1)
                
                # Collect predictions and targets
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
                # Save some example predictions for visualization
                if i == 0:
                    save_prediction_examples(inputs, preds, targets, output_dir, task)
            
            except Exception as e:
                logger.error(f"Error in evaluation batch {i}: {e}")
                continue
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    try:
        if task == 'landcover':
            # For land cover, we calculate metrics for each class
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average=None, zero_division=0
            )
            
            # Calculate IoU (Jaccard index) for each class
            iou = jaccard_score(all_targets, all_preds, average=None, zero_division=0)
            
            # Calculate confusion matrix
            classes = np.unique(np.concatenate((all_targets, all_preds)))
            cm = confusion_matrix(all_targets, all_preds, labels=classes)
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix for Land Cover Classification')
            plt.savefig(os.path.join(output_dir, 'landcover_confusion_matrix.png'))
            
            # Plot metrics by class
            plt.figure(figsize=(15, 5))
            x = np.arange(len(precision))
            width = 0.2
            
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1-score')
            plt.bar(x + 2*width, iou, width, label='IoU')
            
            plt.xlabel('Class')
            plt.ylabel('Score')
            plt.title('Metrics by Class for Land Cover Classification')
            plt.xticks(x, [f'Class {i}' for i in range(len(precision))])
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'landcover_metrics_by_class.png'))
            
            # Print overall metrics
            logger.info(f'Overall Precision: {np.mean(precision):.4f}')
            logger.info(f'Overall Recall: {np.mean(recall):.4f}')
            logger.info(f'Overall F1-score: {np.mean(f1):.4f}')
            logger.info(f'Overall IoU: {np.mean(iou):.4f}')
            
        elif task == 'wildfire':
            # For wildfire, we calculate binary classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='binary', zero_division=0
            )
            
            # Calculate IoU for the positive class (wildfire)
            iou = jaccard_score(all_targets, all_preds, average='binary', zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Fire', 'Fire'], 
                        yticklabels=['No Fire', 'Fire'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix for Wildfire Detection')
            plt.savefig(os.path.join(output_dir, 'wildfire_confusion_matrix.png'))
            
            # Print metrics
            logger.info(f'Precision: {precision:.4f}')
            logger.info(f'Recall: {recall:.4f}')
            logger.info(f'F1-score: {f1:.4f}')
            logger.info(f'IoU: {iou:.4f}')
        
        # Save metrics to file
        with open(os.path.join(output_dir, f'{task}_metrics.txt'), 'w') as f:
            if task == 'landcover':
                f.write(f'Overall Precision: {np.mean(precision):.4f}\n')
                f.write(f'Overall Recall: {np.mean(recall):.4f}\n')
                f.write(f'Overall F1-score: {np.mean(f1):.4f}\n')
                f.write(f'Overall IoU: {np.mean(iou):.4f}\n')
                
                f.write('\nClass-wise metrics:\n')
                for i in range(len(precision)):
                    f.write(f'Class {i}:\n')
                    f.write(f'  Precision: {precision[i]:.4f}\n')
                    f.write(f'  Recall: {recall[i]:.4f}\n')
                    f.write(f'  F1-score: {f1[i]:.4f}\n')
                    f.write(f'  IoU: {iou[i]:.4f}\n')
            elif task == 'wildfire':
                f.write(f'Precision: {precision:.4f}\n')
                f.write(f'Recall: {recall:.4f}\n')
                f.write(f'F1-score: {f1:.4f}\n')
                f.write(f'IoU: {iou:.4f}\n')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'confusion_matrix': cm
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

def save_prediction_examples(inputs, preds, targets, output_dir, task):
    """Save some example predictions for visualization"""
    logger.info("Saving prediction examples...")
    
    try:
        # Convert tensors to numpy arrays
        inputs = inputs.cpu().numpy()
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Create directory for examples if it doesn't exist
        examples_dir = os.path.join(output_dir, 'examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # Save a few examples
        for i in range(min(5, len(inputs))):
            plt.figure(figsize=(15, 5))
            
            # Plot input image
            plt.subplot(1, 3, 1)
            # For RGB images
            if inputs.shape[1] == 3:
                # Denormalize
                img = inputs[i].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
            # For 4-channel images (RGB + NBR)
            elif inputs.shape[1] == 4:
                # Just show RGB part
                img = inputs[i, :3].transpose(1, 2, 0)
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
            plt.title('Input Image')
            plt.axis('off')
            
            # Plot ground truth
            plt.subplot(1, 3, 2)
            if task == 'landcover':
                plt.imshow(targets[i], cmap='tab20')
                plt.title('Ground Truth Land Cover')
            elif task == 'wildfire':
                plt.imshow(targets[i], cmap='hot')
                plt.title('Ground Truth Fire Mask')
            plt.axis('off')
            
            # Plot prediction
            plt.subplot(1, 3, 3)
            if task == 'landcover':
                plt.imshow(preds[i], cmap='tab20')
                plt.title('Predicted Land Cover')
            elif task == 'wildfire':
                plt.imshow(preds[i], cmap='hot')
                plt.title('Predicted Fire Mask')
            plt.axis('off')
            
            plt.savefig(os.path.join(examples_dir, f'{task}_example_{i}.png'))
            plt.close()
        
        logger.info(f"Saved {min(5, len(inputs))} prediction examples to {examples_dir}")
    
    except Exception as e:
        logger.error(f"Error saving prediction examples: {e}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log evaluation configuration
    logger.info(f"Starting evaluation with configuration:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Patch size: {args.patch_size}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Random seed: {args.seed}")
    
    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, task=args.task, batch_size=args.batch_size, patch_size=args.patch_size
        )
        logger.info(f"Test dataloader created with {len(test_loader.dataset)} samples")
        
        # Load model
        model, device = load_model(args)
        
        # Evaluate model
        start_time = time.time()
        metrics = evaluate(model, test_loader, device, args.task, args.output_dir)
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        minutes, seconds = divmod(eval_time, 60)
        logger.info(f"Evaluation completed in {int(minutes)}m {int(seconds)}s")
        
        logger.info(f"Results saved to {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
