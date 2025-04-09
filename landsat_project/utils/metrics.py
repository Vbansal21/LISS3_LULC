import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score
import pandas as pd
import torch

def calculate_segmentation_metrics(true_map, pred_map, num_classes, ignore_index=None):
    """
    Calculates various segmentation metrics.

    Args:
        true_map (np.ndarray): Ground truth segmentation map (flattened or 2D).
        pred_map (np.ndarray): Predicted segmentation map (flattened or 2D).
        num_classes (int): Total number of classes.
        ignore_index (int, optional): Index to ignore in calculations (e.g., background).

    Returns:
        dict: Dictionary containing Overall Accuracy, Mean IoU, and per-class IoU.
    """
    true_flat = true_map.flatten()
    pred_flat = pred_map.flatten()

    if ignore_index is not None:
        valid_mask = true_flat != ignore_index
        true_flat = true_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]

    overall_accuracy = accuracy_score(true_flat, pred_flat)
    
    # Calculate IoU (Jaccard Score)
    labels = list(range(num_classes))
    if ignore_index is not None and ignore_index in labels:
        labels.remove(ignore_index)
        
    iou_per_class = jaccard_score(true_flat, pred_flat, labels=labels, average=None)
    mean_iou = np.nanmean(iou_per_class) # Use nanmean to handle classes not present
    
    metrics = {
        'Overall Accuracy': overall_accuracy,
        'Mean IoU': mean_iou,
        'Per-Class IoU': {label: iou for label, iou in zip(labels, iou_per_class)}
    }
    
    return metrics

def calculate_change_metrics(map1, map2, num_classes):
    """
    Calculates basic change detection metrics: total changed pixels, 
    percentage change, and a transition matrix.

    Args:
        map1 (np.ndarray): Segmentation map at time 1 (flattened or 2D).
        map2 (np.ndarray): Segmentation map at time 2 (flattened or 2D).
        num_classes (int): Total number of classes.

    Returns:
        dict: Dictionary containing change metrics and the transition matrix.
    """
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()
    
    if map1_flat.shape != map2_flat.shape:
        raise ValueError("Input maps must have the same shape.")

    total_pixels = map1_flat.size
    changed_pixels = np.sum(map1_flat != map2_flat)
    percentage_change = (changed_pixels / total_pixels) * 100
    
    # Calculate transition matrix (confusion matrix between map1 and map2)
    # Rows: Class in map1, Columns: Class in map2
    transition_matrix = confusion_matrix(map1_flat, map2_flat, labels=list(range(num_classes)))
    
    # Convert matrix to DataFrame for better readability (optional)
    class_labels = [f'Class_{i}' for i in range(num_classes)]
    transition_df = pd.DataFrame(transition_matrix, index=class_labels, columns=class_labels)
    
    metrics = {
        'Total Pixels': total_pixels,
        'Changed Pixels': changed_pixels,
        'Percentage Change': percentage_change,
        'Transition Matrix': transition_df
    }
    
    return metrics 

class SegmentationMetrics:
    """Calculates common segmentation metrics (Accuracy, Precision, Recall, F1, IoU)."""
    def __init__(self, num_classes: int, ignore_index: int = -100, device: torch.device = torch.device('cpu')):
        """
        Args:
            num_classes (int): Number of target classes.
            ignore_index (int, optional): Specifies a target value that is ignored.
                                         Defaults to -100.
            device (torch.device, optional): Device to store the confusion matrix on.
                                            Defaults to torch.device('cpu').
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.confusion_matrix = torch.zeros((num_classes, num_classes), device=self.device, dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix with new predictions and targets.

        Args:
            pred (torch.Tensor): Predicted segmentation map (usually logits or probabilities),
                               shape (N, C, H, W).
            target (torch.Tensor): Ground truth segmentation map, shape (N, H, W).
        """
        pred = pred.argmax(dim=1).flatten()  # Get predicted class indices
        target = target.flatten()

        # Filter out ignored indices
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        # Ensure pred and target are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)

        # Update confusion matrix: Add counts for pairs (target_class, predicted_class)
        indices = target * self.num_classes + pred
        self.confusion_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute_metrics(self, smooth: float = 1e-6) -> dict:
        """
        Compute segmentation metrics from the confusion matrix.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero.
                                      Defaults to 1e-6.

        Returns:
            dict: Dictionary containing computed metrics:
                  'accuracy', 'iou', 'dice', 
                  'precision_class_<i>', 'recall_class_<i>', 'f1_class_<i>', 'iou_class_<i>' (for each class i),
                  'mean_precision', 'mean_recall', 'mean_f1', 'mean_iou' (macro averages).
        """
        metrics = {}
        conf_matrix = self.confusion_matrix.float()

        # True Positives, False Positives, False Negatives
        tp = conf_matrix.diag()
        fp = conf_matrix.sum(dim=0) - tp
        fn = conf_matrix.sum(dim=1) - tp

        # Overall accuracy
        total_pixels = conf_matrix.sum()
        metrics['accuracy'] = (tp.sum() / (total_pixels + smooth)).item()

        # Intersection over Union (IoU) and Dice Coefficient
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + smooth)
        dice = (2 * intersection) / (union + intersection + smooth)

        metrics['iou'] = iou.mean().item() # Mean IoU (mIoU)
        metrics['dice'] = dice.mean().item() # Mean Dice

        # Per-class metrics
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        for i in range(self.num_classes):
            class_tp = tp[i]
            class_fp = fp[i]
            class_fn = fn[i]
            
            precision = class_tp / (class_tp + class_fp + smooth)
            recall = class_tp / (class_tp + class_fn + smooth)
            f1 = 2 * (precision * recall) / (precision + recall + smooth)
            class_iou = iou[i]
            
            metrics[f'precision_class_{i}'] = precision.item()
            metrics[f'recall_class_{i}'] = recall.item()
            metrics[f'f1_class_{i}'] = f1.item()
            metrics[f'iou_class_{i}'] = class_iou.item()
            
            # Append to lists for mean calculation (only if class has support)
            if (class_tp + class_fn) > 0: # Check if class exists in ground truth
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                iou_list.append(class_iou)

        # Mean (Macro) metrics - average over classes present in the ground truth
        metrics['mean_precision'] = (torch.stack(precision_list).mean() if precision_list else torch.tensor(0.0)).item()
        metrics['mean_recall'] = (torch.stack(recall_list).mean() if recall_list else torch.tensor(0.0)).item()
        metrics['mean_f1'] = (torch.stack(f1_list).mean() if f1_list else torch.tensor(0.0)).item()
        metrics['mean_iou'] = (torch.stack(iou_list).mean() if iou_list else torch.tensor(0.0)).item() # Same as mIoU above, kept for consistency

        return metrics

    def reset(self):
        """Reset the confusion matrix to zeros."""
        self.confusion_matrix.zero_()

# Example Usage:
if __name__ == '__main__':
    # Example: 3 classes, ignore index 2
    metrics_calculator = SegmentationMetrics(num_classes=3, ignore_index=2, device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    # Mock prediction (logits) and target
    # N=2, C=3, H=4, W=4
    mock_pred = torch.randn(2, 3, 4, 4, device=metrics_calculator.device)
    mock_target = torch.randint(0, 3, (2, 4, 4), device=metrics_calculator.device)
    # Introduce some ignored pixels
    mock_target[0, 0, 0] = 2 
    mock_target[1, 2, 3] = 2

    print("Mock Prediction (shape):", mock_pred.shape)
    print("Mock Target (before ignore):\n", mock_target)

    # Update metrics
    metrics_calculator.update(mock_pred, mock_target)
    print("\nConfusion Matrix:\n", metrics_calculator.confusion_matrix)

    # Compute metrics
    results = metrics_calculator.compute_metrics()
    print("\nComputed Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Reset and recompute (should be zeros)
    metrics_calculator.reset()
    print("\nReset Confusion Matrix:\n", metrics_calculator.confusion_matrix)
    results_reset = metrics_calculator.compute_metrics()
    print("\nMetrics after reset:")
    for key, value in results_reset.items():
        print(f"  {key}: {value:.4f}") 