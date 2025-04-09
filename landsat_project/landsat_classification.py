import argparse
import os
import sys
from pathlib import Path
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Tuple, Dict, Any, Optional
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.serialization
from models.model_config import ModelConfig

# Use absolute imports
from config import PATHS, MODEL, TRAINING, DATASET, LOGGING, OUTPUT, VISUALIZATION
from utils.logging import Logger
from utils.metrics import SegmentationMetrics
from datasets.ucmerced_dataset import get_ucmerced_dataloader, get_default_transforms as get_ucmerced_transforms
from datasets.esa_dataset import get_esa_dataloader, get_default_transforms as get_esa_transforms
from models.model_factory import ModelFactory
from models.unet_model import get_intermediate_features
from inference.inference import run_segmentation_inference, load_model
from inference.change_detection import detect_lulc_change

# Initialize logger
logger = Logger(LOGGING)

# --- Helper Functions (Integrated from Trainer) ---

def visualize_segmentation(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor,
                         class_colors: Dict[int, tuple], num_classes: int, save_path: Optional[str] = None) -> None:
    '''Visualize segmentation results'''
    # Convert tensors to numpy
    image = image.cpu().numpy()
    mask = mask.cpu().numpy()
    # Ensure prediction has class dimension before argmax
    if pred.ndim == 3: # (C, H, W)
        pred_labels = pred.argmax(0).cpu().numpy()
    elif pred.ndim == 4: # (B, C, H, W) -> Use first item
        pred_labels = pred[0].argmax(0).cpu().numpy()
    else: # Assume (H, W)
        pred_labels = pred.cpu().numpy()

    # Create color maps
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    pred_rgb = np.zeros((*pred_labels.shape, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        mask_rgb[mask == class_idx] = color
        pred_rgb[pred_labels == class_idx] = color

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show RGB channels of input if available, else first 3 channels
    display_image = np.transpose(image[:3], (1, 2, 0)) if image.shape[0] >= 3 else np.transpose(image[0], (0, 1)) # Handle single channel
     # Normalize if necessary (assuming input range isn't 0-1 or 0-255)
    if display_image.max() > 1.0 and display_image.max() <= 255.0: # Assuming 0-255 range
        display_image = display_image / 255.0
    elif display_image.min() < 0.0 or display_image.max() > 1.0: # Needs normalization
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-6)

    axes[0].imshow(display_image)
    axes[0].set_title('Input Image (First 3 Channels)')
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.debug(f"Saved visualization to {save_path}")
    else:
        plt.show()


def get_intermediate_features(backbone_type: str, backbone: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """
    Extracts intermediate features from the backbone.
    Handles EfficientNet specifically based on typical layer names/structure.
    NOTE: This might need adjustment if using a heavily modified backbone.
    """
    features = []
    if 'efficientnet' in backbone_type.lower():
        # EfficientNet feature extraction logic (common structure)
        # This assumes standard EfficientNet implementation from libraries like torchvision or timm
        if hasattr(backbone, '_conv_stem'):
            x = backbone._conv_stem(x)
            x = backbone._bn0(x)
            x = backbone._swish(x)
            features.append(x) # Stem output

        block_num = 0
        if hasattr(backbone, '_blocks'):
            for block in backbone._blocks:
                x = block(x)
                # Append features after specific block stages (adjust indices based on desired skip connections)
                # Example: Append after blocks 0, 1, 2, 4, 6 for typical UNet skips with EffNetB0/B-like
                # These indices correspond roughly to feature map downscaling points
                if block_num in [0, 1, 2, 4, 6]: # Adjust these indices based on backbone variant and desired skips
                     if block_num < len(backbone._blocks) -1: # Don't add last block output here
                         features.append(x)
                block_num += 1

        # Add the final feature map before the head/pooling if needed by the decoder
        # This often comes after the final block and potentially a conv_head/bn1
        if hasattr(backbone, '_conv_head'):
            x = backbone._conv_head(x) # Apply final conv if exists
        if hasattr(backbone, '_bn1'):
             x = backbone._bn1(x) # Apply final BN if exists
        if hasattr(backbone, '_swish'):
             x = backbone._swish(x) # Apply final activation
        # Append the final output of the backbone's feature extractor part
        features.append(x)

    elif 'resnet' in backbone_type.lower():
         # Add ResNet feature extraction logic here if needed
         logger.warning(f"Intermediate feature extraction for ResNet not fully implemented.")
         # Basic ResNet example (may need refinement based on specific ResNet model)
         x = backbone.conv1(x)
         x = backbone.bn1(x)
         x = backbone.relu(x)
         features.append(x) # after stem
         x = backbone.maxpool(x)
         x = backbone.layer1(x); features.append(x)
         x = backbone.layer2(x); features.append(x)
         x = backbone.layer3(x); features.append(x)
         x = backbone.layer4(x); features.append(x) # Final layer output

    else:
        logger.error(f"Intermediate feature extraction not implemented for backbone type: {backbone_type}")
        # Return empty list or raise error
        return []

    # Ensure features are returned in order from shallow to deep
    # The example above appends in order, but double-check if modifying
    logger.debug(f"Extracted {len(features)} intermediate feature maps with shapes: {[f.shape for f in features]}")
    return features

def get_classification_head(model, device, num_classes_clf):
    """Creates or retrieves the classification head for the model."""
    # Check if the model already has a custom classification head attached
    if hasattr(model, 'classification_head') and model.classification_head is not None:
        return model.classification_head.to(device)

    # Determine the number of input features from the backbone's final layer
    try:
        # For EfficientNet, we know the output features before the final classification layer
        if 'efficientnet' in model.config.backbone_type:
            # Based on EfficientNet feature map sizes before final layer
            feature_map_channels = {
                'efficientnet-b0': 1280,  # Output channels of final block * expansion
                'efficientnet-b1': 1280,
                'efficientnet-b2': 1408,
                'efficientnet-b3': 1536,
                'efficientnet-b4': 1792,
                'efficientnet-b5': 2048,
                'efficientnet-b6': 2304,
                'efficientnet-b7': 2560,
            }
            in_features = feature_map_channels.get(model.config.backbone_type, 1280)  # Default to b0 size
            logger.info(f"Creating classification head for EfficientNet with {in_features} input features and {num_classes_clf} classes.")
            classification_head = nn.Linear(in_features, num_classes_clf)
            return classification_head.to(device)
        else:
            logger.error(f"Unsupported backbone type for classification: {model.config.backbone_type}")
            return None

    except Exception as e:
        logger.error(f"Error determining input features for classification head: {e}. Returning None.")
        return None

def train_segmentation_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[Any],
    device: torch.device,
    epoch_num: int,
    num_epochs: int,
    metrics_calculator: SegmentationMetrics,
    phase: str = 'Train', # 'Train' or 'Val'
    dataset_name: str = 'unknown' # For logging purposes
) -> Tuple[float, Dict[str, float]]:
    """Run one epoch of training or validation for segmentation or classification."""
    if phase == 'Train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    # Reset metrics only if the calculator exists
    if metrics_calculator:
        metrics_calculator.reset() # Reset metrics at the start of epoch

    # For avg max probability calculation (confidence proxy)
    total_max_prob = 0.0
    total_valid_pixels = 0

    # Determine if it's a classification task based on dataset name or model config
    # Simple check based on name for now, could be more robust
    is_classification_task = 'ucmerced' in dataset_name.lower()

    # Initialize classification head only if needed and only ONCE per epoch
    classification_head = None
    if is_classification_task:
        # This assumes UCMerced is always the classification task
        num_classes_clf = MODEL['num_classes']['ucmerced']
        classification_head = get_classification_head(model, device, num_classes_clf)
        if classification_head is None and phase == 'Train': # Only error out if training needs it and it failed
            logger.error("Failed to get/create classification head for UCMerced training.")
            # Decide how to handle this - maybe raise error or return early
            # For now, let loop continue but batches will be skipped
        elif classification_head is None and phase == 'Val':
             logger.error("Failed to get/create classification head for UCMerced validation.")
             # Allow epoch to proceed but batches will be skipped

    pbar = tqdm(loader, desc=f'Epoch {epoch_num+1}/{num_epochs} [{phase} - {dataset_name}]')
    for batch_idx, batch in enumerate(pbar):
         # Handle different dataloader structures
        if len(batch) == 3: # Expect (image, label, class_name) for UCMerced
            inputs, targets, _ = batch
            if not is_classification_task:
                logger.warning(f"Expected 2 elements for segmentation task {dataset_name}, but got 3. Ignoring third element.")
        elif len(batch) == 2: # Expect (image, mask) for segmentation
            inputs, targets = batch
            if is_classification_task:
                 logger.warning(f"Expected 3 elements for classification task {dataset_name}, but got 2. Cannot get class names.")
        else:
            # Log error and skip batch if structure is unexpected
            logger.error(f"Unexpected batch structure: {len(batch)} elements in batch {batch_idx}. Skipping batch.")
            continue # Skip this batch

        # --- Data Preprocessing and Validation ---
        # Check for NaN/Inf in inputs
        if not torch.isfinite(inputs).all():
            num_invalid = (~torch.isfinite(inputs)).sum().item()
            logger.warning(f"Invalid values (NaN/Inf: {num_invalid}) detected in input batch {batch_idx}, replacing with 0.")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for NaN/Inf in targets (only relevant for segmentation masks initially)
        if not is_classification_task and not torch.isfinite(targets).all():
            num_invalid = (~torch.isfinite(targets)).sum().item()
            # Segmentation masks are typically float before conversion, check here
            logger.warning(f"Invalid values (NaN/Inf: {num_invalid}) detected in mask batch {batch_idx}, replacing with ignore_index {DATASET['ignore_index']}.")
            # Replace NaN/Inf with ignore index. Ensure target is float for nan_to_num if necessary.
            targets = torch.nan_to_num(targets.float(), nan=float(DATASET['ignore_index'])).long() # Convert back to long

        # Move to device
        inputs = inputs.to(device)
        # Target type depends on task - ensure long type for loss
        targets = targets.to(device, dtype=torch.long)

        # Check target values are within valid range for the task
        if is_classification_task:
             num_classes_clf = MODEL['num_classes']['ucmerced']
             if not ((targets >= 0) & (targets < num_classes_clf)).all():
                  logger.error(f"Invalid target labels found in classification batch {batch_idx}. Min: {targets.min()}, Max: {targets.max()}, Num Classes: {num_classes_clf}. Skipping batch.")
                  continue
        else: # Segmentation
             num_classes_seg = MODEL['num_classes']['esa'] # Assume ESA if not classification
             valid_mask = (targets >= 0) & (targets < num_classes_seg)
             ignored_mask = targets == DATASET['ignore_index']
             if not (valid_mask | ignored_mask).all():
                  logger.error(f"Invalid target labels found in segmentation batch {batch_idx}. Min: {targets.min()}, Max: {targets.max()}, Num Classes: {num_classes_seg}, Ignore Index: {DATASET['ignore_index']}. Skipping batch.")
                  continue

        # --- Forward Pass and Loss Calculation ---
        if optimizer: optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'Train'):
            loss = torch.tensor(0.0, device=device) # Initialize loss
            outputs = None # Initialize outputs

            try:
                if is_classification_task:
                    # --- Classification Path (UCMerced) ---
                    # Head should already be initialized outside the loop
                    if classification_head is None:
                        logger.error(f"Classification head is None during {phase}, skipping batch {batch_idx}.")
                        continue

                    # 1. Get features (using the implemented function)
                    # Ensure the model passed has the expected 'backbone' attribute
                    if not hasattr(model, 'backbone'):
                         logger.error("Model does not have a 'backbone' attribute. Cannot extract features.")
                         continue
                    encoder_features = get_intermediate_features(model.config.backbone_type, model.backbone, inputs)
                    if not encoder_features: # Handle case where feature extraction fails
                         logger.error("Failed to extract intermediate features. Skipping batch.")
                         continue
                    final_features = encoder_features[-1] # Deepest features
                    # 2. Pool features
                    pooled_features = F.adaptive_avg_pool2d(final_features, (1, 1)).squeeze(-1).squeeze(-1) # [B, C]
                    # 3. Get logits
                    if classification_head is None: # Final check for validation phase
                        logger.error("Classification head is None during validation forward pass. Skipping batch.")
                        continue
                    logits = classification_head(pooled_features) # [B, Num_Classes_Clf]
                    # 4. Calculate classification loss
                    loss = criterion(logits, targets) # CE expects [B, C] and [B]
                    outputs = logits # Store logits for potential metric calculation

                else:
                    # --- Segmentation Path (e.g., ESA) ---
                    outputs = model(inputs) # Get segmentation logits [B, C, H, W]
                    loss = criterion(outputs, targets) # CE expects [B, C, H, W] and [B, H, W]

                    # --- Calculate Confidence Proxy (Avg Max Probability) ---
                    if outputs is not None and targets is not None:
                         with torch.no_grad(): # No need for gradients here
                              probabilities = torch.softmax(outputs, dim=1) # [B, C, H, W]
                              max_probs, _ = torch.max(probabilities, dim=1) # [B, H, W]

                              # Create mask for valid (non-ignored) pixels
                              valid_pixel_mask = (targets != DATASET['ignore_index']) # [B, H, W]

                              # Sum max probabilities for valid pixels
                              batch_max_prob_sum = torch.sum(max_probs[valid_pixel_mask])
                              batch_valid_pixels = valid_pixel_mask.sum()

                              if batch_valid_pixels > 0:
                                   total_max_prob += batch_max_prob_sum.item()
                                   total_valid_pixels += batch_valid_pixels.item()

            except Exception as e:
                 logger.error(f"Error during forward/loss calculation in phase '{phase}', batch {batch_idx}: {e}")
                 # Skip batch or handle error appropriately
                 continue # Simple skip

            # --- Backward Pass and Optimization (Train only) ---
            if phase == 'Train' and optimizer is not None and torch.isfinite(loss):
                try:
                    loss.backward()
                    optimizer.step()
                    if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): # Step OneCycleLR per batch
                        scheduler.step()
                except Exception as e:
                    logger.error(f"Error during backward/step in batch {batch_idx}: {e}")
                    # Potentially skip rest of epoch or stop training if errors persist
                    continue

        # --- Update Loss and Metrics ---
        if torch.isfinite(loss):
             running_loss += loss.item()
        else:
             logger.warning(f"Non-finite loss ({loss.item()}) detected in phase '{phase}', batch {batch_idx}. Loss not accumulated.")

        # Update metrics calculator (only for segmentation task and if calculator exists)
        if not is_classification_task and metrics_calculator and outputs is not None and targets is not None and torch.isfinite(outputs).all():
            try:
                metrics_calculator.update(outputs.detach(), targets)
            except Exception as e:
                 logger.error(f"Error updating metrics in batch {batch_idx}: {e}")
        # else: # Add classification metrics calculation here if needed
            # pass

        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1)
            # Add other live metrics to postfix if desired (e.g., running accuracy)
        })

    # --- Epoch End Calculations ---
    if (batch_idx + 1) == 0: # Handle case where loader was empty or all batches were skipped
         logger.warning(f"Epoch {epoch_num+1}/{num_epochs} [{phase} - {dataset_name}] - No batches processed.")
         return 0.0, {} # Return zero loss and empty metrics

    epoch_loss = running_loss / (batch_idx + 1)

    # Compute final epoch metrics
    if not is_classification_task:
        # Compute segmentation metrics only if calculator exists
        if metrics_calculator:
            try:
                epoch_metrics = metrics_calculator.compute_metrics()
                # Add confidence proxy
                if total_valid_pixels > 0:
                    epoch_metrics['avg_max_prob'] = total_max_prob / total_valid_pixels
                else:
                    epoch_metrics['avg_max_prob'] = 0.0 # Or NaN, or omit
            except Exception as e:
                 logger.error(f"Error computing metrics at end of epoch {epoch_num+1}: {e}")
                 epoch_metrics = {'avg_max_prob': 0.0} # Provide default for confidence
        else:
             epoch_metrics = {'avg_max_prob': 0.0 if total_valid_pixels > 0 else 0.0} # Still provide confidence if calculated
    else:
        epoch_metrics = {} # Placeholder for classification metrics
        # Add final classification metrics calculation here (e.g., accuracy from accumulated logits/targets)

    epoch_metrics['loss'] = epoch_loss

    return epoch_loss, epoch_metrics


def train_segmentation_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    num_epochs: int,
    device: torch.device,
    output_dir: Path,
    class_colors: Dict[int, tuple],
    use_wandb: bool = True,
    stage: str = 'train', # Add stage identifier ('ucmerced', 'esa', etc.)
    dataset_name: str = 'unknown' # Add dataset name
) -> Tuple[Dict[str, float], Dict[str, float]]:
    '''Train a segmentation model (integrated version).'''
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_val_metric = -float('inf') # Use mIoU or other primary metric for saving
    best_epoch = -1
    best_train_metrics = {}
    best_val_metrics = {}

    # Determine task type and num_classes for metrics
    is_classification_task = 'ucmerced' in dataset_name.lower()
    if is_classification_task:
         # For UCMerced (classification), maybe use accuracy or just loss
         num_classes_metric = MODEL['num_classes']['ucmerced']
         primary_metric_name = 'accuracy' # Placeholder - needs calculation
         metrics_calculator = None # No segmentation metrics needed
         logger.info(f"Running UCMerced classification stage. Primary metric: {primary_metric_name} (TBD)")
    else:
         # For ESA (segmentation)
         num_classes_metric = MODEL['num_classes']['esa']
         primary_metric_name = 'mean_iou' # Use mIoU as primary metric
         ignore_idx = DATASET.get('ignore_index', 255)
         metrics_calculator = SegmentationMetrics(num_classes=num_classes_metric, device=device, ignore_index=ignore_idx)
         logger.info(f"Running {dataset_name} segmentation stage. Primary metric: {primary_metric_name}")


    for epoch in range(num_epochs):
        # Reset metrics calculator at the start of each epoch if it exists
        if metrics_calculator:
            metrics_calculator.reset()

        # Training Epoch
        train_loss, train_metrics = train_segmentation_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs,
            metrics_calculator, phase='Train', dataset_name=dataset_name
        )
        # Log combined loss and metrics
        log_string_train = f"Epoch {epoch+1}/{num_epochs} [Train] Loss: {train_loss:.4f}"
        for name, val in train_metrics.items(): log_string_train += f", {name}: {val:.4f}"
        logger.info(log_string_train)

        # Validation Epoch
        # Reset metrics calculator for validation if it exists
        if metrics_calculator:
            metrics_calculator.reset()

        val_loss, val_metrics = train_segmentation_epoch(
            model, val_loader, criterion, None, None, device, epoch, num_epochs, # No optimizer/scheduler step in val
            metrics_calculator, phase='Val', dataset_name=dataset_name
        )
        # Log combined loss and metrics
        log_string_val = f"Epoch {epoch+1}/{num_epochs} [Val] Loss: {val_loss:.4f}"
        for name, val in val_metrics.items(): log_string_val += f", {name}: {val:.4f}"
        logger.info(log_string_val)


        # Step LR scheduler (if not OneCycleLR)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                  # Use validation loss or primary metric for ReduceLROnPlateau
                  metric_for_scheduler = val_metrics.get(primary_metric_name, val_loss)
                  scheduler.step(metric_for_scheduler)
             else:
                  scheduler.step()

        # Log metrics to W&B (if enabled)
        if use_wandb:
            try:
                 log_data = {
                     f'{stage}/{dataset_name}/train_loss': train_loss,
                     f'{stage}/{dataset_name}/val_loss': val_loss,
                     'epoch': epoch + 1,
                     'learning_rate': optimizer.param_groups[0]['lr']
                 }
                 # Merge metric dicts, prefixing keys
                 for k, v in train_metrics.items(): log_data[f'{stage}/{dataset_name}/train_{k}'] = v
                 for k, v in val_metrics.items(): log_data[f'{stage}/{dataset_name}/val_{k}'] = v
                 wandb.log(log_data, step=epoch)
            except Exception as e:
                 logger.error(f"Failed to log metrics to wandb for epoch {epoch+1}: {e}")


        # Save model checkpoint logic
        # Determine the current metric value to check for improvement
        # For classification, this needs to be calculated if primary_metric_name is not 'loss'
        if is_classification_task:
            # Currently, we don't calculate classification accuracy/metrics in the loop
            # So, we default to saving based on lowest validation loss
            current_metric_val = -val_loss # Use negative loss (higher is better)
            metric_to_beat = -best_val_metric if best_epoch != -1 else -float('inf') # Compare negative losses
            is_better = current_metric_val > metric_to_beat
        else:
            # For segmentation, use the primary metric (e.g., mIoU)
            current_metric_val = val_metrics.get(primary_metric_name, -float('inf')) # Default to -inf if metric missing
            metric_to_beat = best_val_metric
            is_better = current_metric_val > metric_to_beat

        if is_better:
            best_val_metric = current_metric_val if is_classification_task else current_metric_val # Store the actual metric value
            best_epoch = epoch + 1
            best_train_metrics = train_metrics.copy()
            best_val_metrics = val_metrics.copy()

            # Save model state (including config)
            save_path = output_dir / f'{stage}_{dataset_name}_best_model.pth'
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': getattr(model, 'config', None), # Save model config if attached
                'best_val_metric': best_val_metric, # Save the metric value that triggered the save
                'primary_metric_name': primary_metric_name # Save the name of the metric used
            }
            try:
                torch.save(checkpoint, save_path)
                logger.info(f"Saved best model to {save_path} (Epoch: {best_epoch}, Val {primary_metric_name if not is_classification_task else 'Loss'}: {abs(best_val_metric):.4f})")
            except Exception as e:
                 logger.error(f"Failed to save checkpoint for epoch {epoch+1}: {e}")

            # Optional: Log as W&B artifact
            # if use_wandb and OUTPUT.get('log_artifacts', False):
            #     pass # Add W&B artifact logging here

        # Save last model checkpoint (optional)
        if OUTPUT.get('save_last', False):
             # Similar save logic as above, but always save
             pass # Add logic to save last checkpoint if enabled


        # Visualize results (only for segmentation task)
        # Check if visualization is enabled and it's not a classification task
        if not is_classification_task and \
           VISUALIZATION.get('log_visualizations', False) and \
           (epoch % VISUALIZATION.get('vis_freq', 5) == 0 or epoch == num_epochs - 1): # Also vis last epoch
             try:
                  # Get a sample batch from validation loader
                  vis_batch = next(iter(val_loader))
                  # Ensure correct unpacking for segmentation
                  if len(vis_batch) == 2:
                       inputs_vis, masks_vis = vis_batch
                  else:
                       logger.warning(f"Unexpected batch structure for visualization: {len(vis_batch)} elements. Skipping visualization.")
                       continue

                  inputs_vis, masks_vis = inputs_vis.to(device), masks_vis.to(device, dtype=torch.long)

                  model.eval()
                  with torch.no_grad():
                       preds_vis = model(inputs_vis) # [B, C, H, W]

                  # Visualize first item in the batch
                  save_vis_path = vis_dir / f"{stage}_{dataset_name}_epoch_{epoch+1}_vis.png"

                  # Use the actual number of classes for visualization
                  visualize_segmentation(
                       inputs_vis[0], masks_vis[0], preds_vis[0],
                       class_colors, num_classes=num_classes_metric, save_path=str(save_vis_path)
                  )
                  if use_wandb:
                       try:
                            wandb.log({f"{stage}/{dataset_name}/validation_samples": wandb.Image(str(save_vis_path))}, step=epoch)
                       except Exception as e:
                            logger.error(f"Failed to log visualization to wandb: {e}")

             except StopIteration:
                  logger.warning(f"Validation loader empty, cannot generate visualization for epoch {epoch+1}.")
             except Exception as e:
                  logger.warning(f"Failed to generate visualization for epoch {epoch+1}: {e}")


    logger.info(f"Training finished for {stage} - {dataset_name}.")
    logger.info(f"Best Epoch: {best_epoch}, Best Val {primary_metric_name if not is_classification_task else 'Loss'}: {abs(best_val_metric):.4f}")

    # Return the metrics from the best epoch
    return best_train_metrics, best_val_metrics


# --- Main Execution Logic ---

def create_model_config(args, dataset: str = None, num_classes=None, in_channels=None) -> ModelConfig:
    '''Create model configuration based on args, config file, and dynamic values.'''
    # Determine default num_classes and in_channels based on dataset or final default (ESA)
    default_num_classes = MODEL['num_classes'].get(dataset) if dataset else MODEL['num_classes']['esa']
    default_in_channels = MODEL['in_channels'].get(dataset) if dataset else MODEL['in_channels']['esa']

    config_dict = {
        'model_type': args.model_type or MODEL['model_type'],
        'num_classes': num_classes or default_num_classes,
        'in_channels': in_channels or default_in_channels,
        'backbone_type': args.backbone_type or MODEL['backbone_type'],
        'use_pretrained_backbone': args.pretrained if args.pretrained is not None else MODEL['use_pretrained_backbone'],
        'freeze_backbone': MODEL.get('freeze_backbone', False), # Use .get for safety
        'decoder_channels': MODEL.get('decoder_channels', [256, 128, 64, 32, 16]), # Use .get
        'use_skip_connections': MODEL.get('use_skip_connections', True),
        'dropout_rate': MODEL.get('dropout_rate', 0.1),
    }
    logger.info(f"Creating model config for '{dataset or 'default'}': { {k:v for k,v in config_dict.items()} }") # Log key params
    return ModelConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Land Use Land Cover Segmentation & Change Detection")
    parser.add_argument('--mode', choices=['train_inference', 'change_detect'], default='train_inference',
                      help='Mode: train_inference (train UCM->ESA then infer) or change_detect')
    parser.add_argument('--model_type', choices=['unet'], default=MODEL['model_type'], # Default from config
                      help='Type of model architecture to use')
    parser.add_argument('--backbone_type', choices=['efficientnet-b0', 'resnet50'], default=MODEL['backbone_type'], # Default from config
                      help='Type of backbone for UNet')
    parser.add_argument('--landsat_path_t1', type=str, default=str(PATHS['landsat']),
                      help='Path to the Landsat scene directory for inference/change_detect')
    parser.add_argument('--landsat_path_t2', type=str, 
                      help='Path to the second Landsat image (for change_detect mode)')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT['output_dir']),
                      help='Directory to save outputs')
    parser.add_argument('--checkpoint_path', type=str, # No default, loaded/saved during run
                      help='Path to a specific model checkpoint to load (optional, overrides training or used for change_detect)')
    parser.add_argument('--test_mode', action='store_true',
                      help='Run in test mode (e.g., fewer samples, smaller dataset portions)')
    parser.add_argument('--pretrained', action='store_true', default=None, # Default to None, let config handle it unless specified
                      help='Force use/non-use of pretrained backbone weights (overrides config)')
    parser.add_argument('--no_wandb', action='store_true', default=False,
                      help='Disable Weights & Biases logging')

    # Add specific training args if needed, overriding config.py
    parser.add_argument('--ucmerced_epochs', type=int, default=TRAINING['num_epochs']['ucmerced'], help='Epochs for UCMerced training')
    parser.add_argument('--esa_epochs', type=int, default=TRAINING['num_epochs']['esa'], help='Epochs for ESA training')
    parser.add_argument('--batch_size', type=int, default=None, help='Override default batch size (same for both stages)')
    parser.add_argument('--lr', type=float, default=TRAINING['optimizer']['lr'], help='Initial learning rate for AdamW')
    parser.add_argument('--load_ucmerced_from', type=str, default=None,
                        help='Path to a specific UCMerced checkpoint to load before ESA training (overrides default behavior if ucmerced_epochs=0)')


    args = parser.parse_args()

    # Set device
    device = torch.device(TRAINING['device'])
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_dir / 'models' # Specific dir for models
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Handle W&B initialization centrally if needed and not disabled
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            # Check if run already active (e.g., from previous call)
             if wandb.run is None:
                wandb.init(project=LOGGING['wandb']['project'], entity=LOGGING['wandb'].get('entity'), config=vars(args))
                logger.info("Wandb initialized.")
             else:
                 logger.info("Wandb already initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}. Disabling wandb.")
            use_wandb = False


    # --- Execute selected mode --- 
    if args.mode == 'train_inference':
        logger.info("Starting Train & Inference Mode")

        # Initialize checkpoint path in case UCMerced is skipped
        ucmerced_checkpoint_path = None

        # --- Stage 1: Train on UCMerced ---
        if args.ucmerced_epochs > 0:
            logger.info("="*20 + " Stage 1: Training on UCMerced " + "="*20)
            ucmerced_output_dir = model_output_dir / 'ucmerced'

            # Get UCMerced transforms
            ucmerced_transforms_dict = get_ucmerced_transforms(
                input_size=DATASET['patch_size'],
                mean=DATASET['normalization']['ucmerced']['mean'],
                std=DATASET['normalization']['ucmerced']['std'],
            )

            # Create model for UCMerced
            model_config_ucmerced = create_model_config(args, 'ucmerced')
            model_ucmerced = ModelFactory.create_model(model_config_ucmerced)
            model_ucmerced.config = model_config_ucmerced # Attach config for saving
            model_ucmerced = model_ucmerced.to(device)

            # Get UCMerced dataloaders
            ucm_batch_size = args.batch_size or TRAINING['batch_size']['ucmerced']
            train_loader_ucm = get_ucmerced_dataloader(
                root_dir=PATHS['uc_merced'], 
                batch_size=ucm_batch_size, 
                transform=ucmerced_transforms_dict.get('train_transform'),
                split='train',
                num_workers=TRAINING.get('num_workers', 4),
                val_split=0.2
            )
            val_loader_ucm = get_ucmerced_dataloader(
                root_dir=PATHS['uc_merced'], 
                batch_size=ucm_batch_size, 
                transform=ucmerced_transforms_dict.get('val_transform'),
                split='val',
                num_workers=TRAINING.get('num_workers', 4),
                val_split=0.2
            )

            # Training setup for UCMerced
            criterion_ucm = nn.CrossEntropyLoss(ignore_index=DATASET.get('ignore_index', 255)) # Use .get
            optimizer_ucm = optim.AdamW(model_ucmerced.parameters(), lr=args.lr, weight_decay=TRAINING['optimizer'].get('weight_decay', 0.01)) # Use .get
            # Scheduler (Example: OneCycleLR)
            scheduler_ucm = optim.lr_scheduler.OneCycleLR(
                 optimizer_ucm, max_lr=args.lr, epochs=args.ucmerced_epochs, steps_per_epoch=len(train_loader_ucm),
                 pct_start=TRAINING['scheduler'].get('pct_start', 0.3), # Use .get
                 div_factor=TRAINING['scheduler'].get('div_factor', 25.0),
                 final_div_factor=TRAINING['scheduler'].get('final_div_factor', 1e4)
            )

            # Train on UCMerced
            train_segmentation_model(
                model=model_ucmerced, train_loader=train_loader_ucm, val_loader=val_loader_ucm,
                criterion=criterion_ucm, optimizer=optimizer_ucm, scheduler=scheduler_ucm,
                num_epochs=args.ucmerced_epochs, device=device, output_dir=ucmerced_output_dir,
                class_colors=VISUALIZATION['class_colors'].get('ucmerced', {}), # Use .get
                use_wandb=use_wandb, stage='ucmerced', dataset_name='ucmerced'
            )
            logger.info("UCMerced training completed.")
            # Define the expected path to the saved best UCMerced model
            # Ensure this matches the filename format used in train_segmentation_model
            # Hardcode the filename as stage/dataset names are not in DATASET config dict
            ucmerced_checkpoint_path = ucmerced_output_dir / "ucmerced_ucmerced_best_model.pth"
        else:
            logger.info("Skipping UCMerced training stage as epochs is 0.")

        # --- Stage 2: Train on ESA ---
        logger.info("="*20 + " Stage 2: Training on ESA " + "="*20)
        esa_output_dir = model_output_dir / 'esa'
        ucmerced_best_model_path = model_output_dir / 'ucmerced' / 'ucmerced_ucmerced_best_model.pth' # Define standard path

        # Get ESA transforms - REMOVED as normalization is internal
        # esa_transforms = get_esa_transforms(...)

        # Create/Load model for ESA
        model_config_esa = create_model_config(args, 'esa')
        model_esa = ModelFactory.create_model(model_config_esa)
        model_esa.config = model_config_esa # Attach config

        # Determine which UCMerced checkpoint to load, if any
        ucm_checkpoint_to_load = None
        if args.load_ucmerced_from:
            load_path = Path(args.load_ucmerced_from)
            if load_path.exists():
                ucm_checkpoint_to_load = load_path
                logger.info(f"Explicitly loading UCMerced weights from: {ucm_checkpoint_to_load}")
            else:
                logger.warning(f"Specified UCMerced checkpoint not found: {args.load_ucmerced_from}. Will proceed without it.")
        elif args.ucmerced_epochs > 0 and ucmerced_best_model_path.exists():
             ucm_checkpoint_to_load = ucmerced_best_model_path
             logger.info(f"Found best UCMerced model from training stage: {ucm_checkpoint_to_load}")


        # Load weights from UCMerced checkpoint if a path was determined
        loaded_from_ucm = False
        if ucm_checkpoint_to_load: # Check if we have a valid path to load from
            try:
                logger.info(f"Attempting to load UCMerced weights from: {ucm_checkpoint_to_load}")

                # Ensure ModelConfig is safe for loading *before* the torch.load call
                try:
                    current_globals = torch.serialization.get_safe_globals()
                    if ModelConfig not in current_globals:
                        torch.serialization.add_safe_globals([ModelConfig])
                        logger.debug("Added ModelConfig to torch safe globals for loading.")
                except AttributeError:
                    logger.warning("torch.serialization safety functions not found (likely older PyTorch). Loading without explicit safety.")
                except Exception as e:
                    logger.warning(f"Error adding ModelConfig to torch safe globals: {e}. Proceeding with load.")


                # Load checkpoint with weights_only=False as we expect ModelConfig object
                checkpoint = torch.load(ucm_checkpoint_to_load, map_location=device, weights_only=False) # Set weights_only=False

                # ... rest of the loading logic ...
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else: # Assume the checkpoint is just the state dict
                     pretrained_dict = checkpoint

                model_dict = model_esa.state_dict()

                # Filter incompatible layers (e.g., different num_classes in final layer)
                pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                missing_keys, unexpected_keys = model_esa.load_state_dict(pretrained_dict_filtered, strict=False) # Use filtered dict
                logger.info(f"Loaded ESA model weights from UCMerced. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                # Optionally log the keys for debugging
                # if unexpected_keys: logger.debug(f"Unexpected keys: {unexpected_keys}")
                # if missing_keys: logger.debug(f"Missing keys: {missing_keys}")
                loaded_from_ucm = True
            except Exception as e:
                logger.error(f"Failed to load UCMerced checkpoint {ucm_checkpoint_to_load}: {e}. Training ESA from scratch/pretrained.")
        else:
            logger.warning("No UCMerced checkpoint specified or found. Training ESA model from scratch or specified pretrained weights.")


        model_esa = model_esa.to(device)

        # Get ESA dataloaders (pass transform=None)
        esa_batch_size = args.batch_size or TRAINING['batch_size'].get('esa', 16) # Use .get
        train_loader_esa = get_esa_dataloader(
            esa_directory=PATHS['esa_worldcover'], landsat_path=PATHS['landsat'], batch_size=esa_batch_size,
            transform=None, # Pass None, normalization is internal to dataset
            num_samples=DATASET['num_samples'] if not args.test_mode else 100,
            test_mode=args.test_mode, num_workers=TRAINING.get('num_workers', 4)
        )
        val_loader_esa = get_esa_dataloader(
            esa_directory=PATHS['esa_worldcover'], landsat_path=PATHS['landsat'], batch_size=esa_batch_size,
            transform=None, # Pass None, normalization is internal to dataset
            num_samples=DATASET['num_samples'] // 5 if not args.test_mode else 50,
            test_mode=args.test_mode, num_workers=TRAINING.get('num_workers', 4)
        )

        # Training setup for ESA (fine-tuning)
        criterion_esa = nn.CrossEntropyLoss(ignore_index=DATASET.get('ignore_index', 255))
        # Use a lower LR for fine-tuning if loaded from UCM, else use base LR
        fine_tune_lr_multiplier = 0.1 if loaded_from_ucm else 1.0
        fine_tune_lr = args.lr * fine_tune_lr_multiplier
        logger.info(f"Using fine-tune LR: {fine_tune_lr} (Multiplier: {fine_tune_lr_multiplier})")
        optimizer_esa = optim.AdamW(model_esa.parameters(), lr=fine_tune_lr, weight_decay=TRAINING['optimizer'].get('weight_decay', 0.01))
        # Only initialize scheduler if epochs > 0
        scheduler_esa = None
        if args.esa_epochs > 0:
             if not train_loader_esa: # Check if loader is valid
                  logger.error("ESA train loader is not initialized. Cannot create scheduler.")
             else:
                  scheduler_esa = optim.lr_scheduler.OneCycleLR(
                      optimizer_esa, max_lr=fine_tune_lr, epochs=args.esa_epochs, steps_per_epoch=len(train_loader_esa),
                      pct_start=TRAINING['scheduler'].get('pct_start', 0.3),
                      div_factor=TRAINING['scheduler'].get('div_factor', 25.0),
                      final_div_factor=TRAINING['scheduler'].get('final_div_factor', 1e4)
                  )
                  logger.info(f"Initialized OneCycleLR scheduler for ESA stage ({args.esa_epochs} epochs).")
        else:
             logger.info("Skipping ESA scheduler initialization as esa_epochs is 0.")


        # Train on ESA
        if args.esa_epochs > 0:
             train_segmentation_model(
                 model=model_esa, train_loader=train_loader_esa, val_loader=val_loader_esa,
                 criterion=criterion_esa, optimizer=optimizer_esa, scheduler=scheduler_esa,
                 num_epochs=args.esa_epochs, device=device, output_dir=esa_output_dir,
                 class_colors=VISUALIZATION['class_colors'].get('esa', {}),
                 use_wandb=use_wandb, stage='esa', dataset_name='esa'
             )
             logger.info("ESA training completed.")
             # Correctly define the expected checkpoint path based on saving logic
             esa_checkpoint_path = esa_output_dir / f"esa_esa_best_model.pth"
        else:
             logger.info("Skipping ESA training stage as esa_epochs is 0.")
             esa_checkpoint_path = None # No checkpoint saved if training skipped


        # --- Stage 3: Inference ---
        logger.info("="*20 + " Stage 3: Performing Inference " + "="*20)
        inference_output_dir = output_dir / 'inference'
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        # Determine checkpoint to use for inference (prefer ESA best, fallback to user provided or UCM best)
        inference_checkpoint_path = None
        if args.checkpoint_path and Path(args.checkpoint_path).exists():
            inference_checkpoint_path = Path(args.checkpoint_path)
            logger.info(f"Using user-provided checkpoint for inference: {inference_checkpoint_path}")
        elif esa_checkpoint_path and esa_checkpoint_path.exists(): # Check if not None *and* exists
            inference_checkpoint_path = esa_checkpoint_path
            logger.info(f"Using best ESA model checkpoint for inference: {inference_checkpoint_path}")
        elif ucmerced_checkpoint_path and ucmerced_checkpoint_path.exists(): # Also check if ucmerced_checkpoint_path is not None
             inference_checkpoint_path = ucmerced_checkpoint_path
             logger.warning(f"ESA checkpoint not found, using best UCMerced model for inference: {inference_checkpoint_path}")
        else:
             logger.error("No suitable checkpoint found for inference. Please provide one via --checkpoint_path or ensure training stages completed.")
             sys.exit(1)


        logger.info(f"Loading model for inference from: {inference_checkpoint_path}")
        inference_model = None
        inference_model_config = None # Initialize
        # Load model using config from checkpoint if possible
        try:
            # Allowlist ModelConfig for safe loading
            # This is needed if torch.load defaults to weights_only=True and config is saved
            try:
                 current_globals = torch.serialization.get_safe_globals()
                 if ModelConfig not in current_globals:
                      torch.serialization.add_safe_globals([ModelConfig])
                      logger.debug("Added ModelConfig to torch safe globals for loading.")
            except AttributeError:
                 logger.warning("torch.serialization safety functions not found (likely older PyTorch). Loading without explicit safety.")
            except Exception as e:
                 logger.warning(f"Error handling torch safe globals: {e}. Proceeding with load.")

            checkpoint_data = torch.load(inference_checkpoint_path, map_location=device)

            # Determine ModelConfig for inference model
            loaded_config_data = checkpoint_data.get('config') if isinstance(checkpoint_data, dict) else None # Check if checkpoint is dict
            if isinstance(loaded_config_data, ModelConfig):
                logger.info("Loading model config from checkpoint (ModelConfig object).")
                inference_model_config = loaded_config_data
            elif isinstance(loaded_config_data, dict):
                logger.info("Loading model config from checkpoint (dict), recreating ModelConfig.")
                # Recreate ModelConfig, assuming ESA context if dict is partial
                # Prioritize values from the loaded dict
                base_config_dict = create_model_config(args, 'esa').__dict__ # Get defaults for ESA
                base_config_dict.update(loaded_config_data) # Update with loaded values
                try:
                    inference_model_config = ModelConfig(**base_config_dict)
                except TypeError as te:
                    logger.error(f"Error creating ModelConfig from loaded dict: {te}. Missing keys or invalid values in {loaded_config_data}")
                    sys.exit(1) # Exit if config cannot be created
            else:
                logger.warning("Model config not found or in unexpected format in checkpoint. Creating default ESA config for inference.")
                inference_model_config = create_model_config(args, 'esa') # Fallback to default ESA config

            # Create the model using the determined config
            inference_model = ModelFactory.create_model(inference_model_config)
            inference_model = inference_model.to(device)

            # Load state dict
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                 model_state = checkpoint_data['model_state_dict']
            # Handle case where checkpoint *is* the state dict (e.g., from older saving method or direct save)
            elif isinstance(checkpoint_data, dict) and all(isinstance(k, str) for k in checkpoint_data.keys()):
                 logger.info("Checkpoint appears to be a model state_dict directly.")
                 model_state = checkpoint_data
            else:
                 # This case might occur if loading failed or format is unexpected
                 logger.error("Could not extract model state_dict from checkpoint. Checkpoint format might be invalid or loading failed.")
                 sys.exit(1) # Exit if state dict cannot be determined

            missing_keys, unexpected_keys = inference_model.load_state_dict(model_state, strict=False)
            logger.info(f"Loaded inference model state dict. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            # if unexpected_keys: logger.debug(f"Unexpected keys: {unexpected_keys}")
            # if missing_keys: logger.debug(f"Missing keys: {missing_keys}")

            inference_model.eval() # Set model to evaluation mode

        except FileNotFoundError:
            logger.error(f"Inference checkpoint file not found: {inference_checkpoint_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load model for inference from {inference_checkpoint_path}: {e}")
            sys.exit(1)


        # Ensure model was loaded successfully
        if inference_model is None:
             logger.error("Model could not be loaded for inference. Exiting.")
             sys.exit(1)


        # Perform inference
        logger.info(f"Starting inference on: {args.landsat_path_t1}")
        try:
            # Ensure the input path exists before calling inference
            landsat_input_path = Path(args.landsat_path_t1)
            if not landsat_input_path.exists():
                logger.error(f"Landsat input path for inference does not exist: {landsat_input_path}")
                sys.exit(1)

            # Determine inference batch size from config or args
            inference_batch_size = args.batch_size or TRAINING['batch_size'].get('inference', 1)
            logger.info(f"Using inference batch size: {inference_batch_size}")

            # Use the updated function signature for run_inference
            run_segmentation_inference( # Use the correct imported name
                model=inference_model,
                landsat_scene_path=str(landsat_input_path), # Pass scene path string
                output_dir=inference_output_dir,      # Pass output directory Path obj
                config=inference_model_config,         # Pass the loaded/created config
                device=device,
                test_mode=args.test_mode,
                batch_size=inference_batch_size      # Pass determined batch size
            )
            logger.info(f"Inference completed. Results saved to: {inference_output_dir}")
        except SyntaxError as se:
            # This might indicate issues reading metadata or config files within run_inference
            logger.error(f"A SyntaxError occurred during inference, potentially reading an input file: {se}") # Removed exc_info
            logger.error("Please check the format and contents of the files being processed by the inference step, especially metadata or auxillary files.")
            sys.exit(1)
        except FileNotFoundError as fnf:
            logger.error(f"A required file was not found during inference: {fnf}") # Removed exc_info
            logger.error(f"Please ensure the input path '{args.landsat_path_t1}' and all required data files (e.g., TIFs, metadata) exist and are accessible.")
            sys.exit(1)
        except ImportError as ie:
             logger.error(f"An ImportError occurred during inference: {ie}. Missing dependency?") # Removed exc_info
             sys.exit(1)
        except ValueError as ve:
            # Catch potential value errors from data processing/shape mismatches
            logger.error(f"A ValueError occurred during inference: {ve}") # Removed exc_info
            logger.error("This could be due to unexpected data values, shape mismatches, or configuration issues.")
            sys.exit(1)
        except RuntimeError as rte:
             # Catch potential CUDA/torch runtime errors
            logger.error(f"A PyTorch RuntimeError occurred during inference: {rte}") # Removed exc_info
            logger.error("Check for CUDA memory issues, tensor device mismatches, or invalid operations.")
            sys.exit(1)
        except Exception as e:
            # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred during inference: {e}") # Removed exc_info
            sys.exit(1)


    elif args.mode == 'change_detect':
        logger.info("Starting Change Detection Mode")
        if not args.checkpoint_path:
            logger.error("Checkpoint path (--checkpoint_path) is required for change detection mode")
            sys.exit(1)
        if not args.landsat_path_t1 or not args.landsat_path_t2:
            logger.error("Both --landsat_path_t1 and --landsat_path_t2 are required for change detection")
            sys.exit(1)
        if not Path(args.checkpoint_path).exists():
             logger.error(f"Change detection checkpoint not found: {args.checkpoint_path}")
             sys.exit(1)


        change_output_dir = output_dir / 'change_detection'
        change_output_dir.mkdir(parents=True, exist_ok=True)

        # Load model (using the same checkpoint for both images)
        try:
            logger.info(f"Loading model for change detection from: {args.checkpoint_path}")
            # Reuse inference loading logic if applicable, or use simpler load_model
            # For simplicity here, using the imported load_model, assuming it handles config correctly
            # If load_model needs explicit config, adapt the inference loading logic
            model = load_model(args.checkpoint_path, device)
            logger.info("Model loaded successfully for change detection.")
        except Exception as e:
             logger.error(f"Failed to load model for change detection from {args.checkpoint_path}: {e}")
             sys.exit(1)


        output_change_map_path = change_output_dir / "change_map.tif"
        output_json_path = change_output_dir / "change_summary.json"

        # Assuming change detection uses ESA classes
        class_labels_esa = VISUALIZATION['class_colors'].get('esa', {})
        if not class_labels_esa:
            logger.warning("ESA class color mapping not found in config. Change summary might be affected.")


        logger.info(f"Running change detection between {args.landsat_path_t1} and {args.landsat_path_t2}")
        logger.info(f"Saving change map to: {output_change_map_path}")
        try:
            detect_lulc_change(
                model=model,
                    landsat_t1_dir=args.landsat_path_t1,
                    landsat_t2_dir=args.landsat_path_t2,
                    output_change_map_path=str(output_change_map_path),
                    output_json_path=str(output_json_path),
                device=device,
                    class_labels=class_labels_esa # Pass class labels/colors if needed by function
                )
            logger.info(f"Change detection completed. Map saved to {output_change_map_path}, Summary saved to {output_json_path}")
        except FileNotFoundError as e:
             logger.error(f"Error during change detection - file not found: {e}. Check scene directory structures.")
             sys.exit(1)
        except Exception as e:
             logger.error(f"An unexpected error occurred during change detection: {e}")
             sys.exit(1)


    else:
        logger.error(f"Invalid mode selected: {args.mode}")
        sys.exit(1)

    # Finish wandb run if active
    if use_wandb and wandb.run is not None:
        try:
            wandb.finish()
            logger.info("Wandb run finished.")
        except Exception as e:
            logger.error(f"Failed to properly finish wandb run: {e}")


    logger.info("Script finished.")


if __name__ == "__main__":
    main()