import logging
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import psutil
import torch.cuda as cuda

class MemoryManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_memory_usage(self):
        if self.device.type == 'cuda':
            return {
                'gpu_allocated': cuda.memory_allocated() / 1024**3,
                'gpu_cached': cuda.memory_reserved() / 1024**3,
                'gpu_free': cuda.memory_reserved() - cuda.memory_allocated() / 1024**3
            }
        else:
            process = psutil.Process()
            return {
                'cpu_memory': process.memory_info().rss / 1024**3
            }
            
    def clear_memory(self):
        if self.device.type == 'cuda':
            cuda.empty_cache()
        gc.collect()

class MetricsTracker:
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list
        self.reset()
        self.memory_manager = MemoryManager()
        
    def reset(self):
        self.metrics = {metric: [] for metric in self.metrics_list}
        self.memory_usage = []
        
    def update(self, **kwargs):
        for metric, value in kwargs.items():
            if metric in self.metrics:
                self.metrics[metric].append(value)
        self.memory_usage.append(self.memory_manager.get_memory_usage())
        
    def get_average(self, metric):
        if metric in self.metrics and self.metrics[metric]:
            return np.mean(self.metrics[metric])
        return 0.0
    
    def get_summary(self):
        return {metric: self.get_average(metric) for metric in self.metrics_list}
        
    def plot_memory_usage(self, save_path):
        if not self.memory_usage:
            return
            
        plt.figure(figsize=(10, 6))
        if 'gpu_allocated' in self.memory_usage[0]:
            plt.plot([m['gpu_allocated'] for m in self.memory_usage], label='GPU Allocated')
            plt.plot([m['gpu_cached'] for m in self.memory_usage], label='GPU Cached')
            plt.ylabel('Memory (GB)')
        else:
            plt.plot([m['cpu_memory'] for m in self.memory_usage], label='CPU Memory')
            plt.ylabel('Memory (GB)')
            
        plt.xlabel('Step')
        plt.title('Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

class Logger:
    def __init__(self, config):
        self.config = config
        self.setup_file_logging()
        self.setup_tensorboard()
        self.metrics_tracker = MetricsTracker(config.get('metrics', []))
        self.memory_manager = MemoryManager()
        
    def setup_file_logging(self):
        """Setup file logging with the specified configuration"""
        # Create log directory if it doesn't exist
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging configuration
        logging.basicConfig(
            level=getattr(logging, self.config['level']),
            format=self.config['format'],
            handlers=[
                logging.FileHandler(self.config['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LULC')
        
    def setup_tensorboard(self):
        """Setup TensorBoard logging if enabled"""
        if self.config['tensorboard']['enabled']:
            # Create tensorboard directory
            tensorboard_dir = Path(self.config['tensorboard']['log_dir'])
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            
            # Create run directory with timestamp
            run_dir = tensorboard_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir.mkdir(parents=True, exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(log_dir=str(run_dir))
        else:
            self.tensorboard_writer = None
            
    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)
        
    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)
        
    def error(self, msg):
        """Log error message"""
        self.logger.error(msg)
        
    def debug(self, msg):
        """Log debug message"""
        self.logger.debug(msg)
        
    def log_metrics(self, phase, metrics_dict, epoch):
        """Log metrics to both file and TensorBoard"""
        # Log to file
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
        self.logger.info(f"{phase.capitalize()} - Epoch {epoch} - {metrics_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for metric, value in metrics_dict.items():
                self.tensorboard_writer.add_scalar(f"{phase}/{metric}", value, epoch)
                
    def log_confusion_matrix(self, y_true, y_pred, class_names, phase, epoch):
        """Log confusion matrix to both file and TensorBoard"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Confusion Matrix - {phase.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        cm_path = self.config['tensorboard']['log_dir'] / 'confusion_matrices'
        cm_path.mkdir(exist_ok=True)
        plt.savefig(cm_path / f'{phase}_epoch_{epoch}.png')
        plt.close()
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(
                f"{phase}/confusion_matrix",
                plt.gcf(),
                epoch
            )
            
    def log_class_distribution(self, y_true, class_names, phase, epoch):
        """Log class distribution to both file and TensorBoard"""
        unique, counts = np.unique(y_true, return_counts=True)
        distribution = dict(zip([class_names[i] for i in unique], counts))
        
        # Create and save distribution plot
        plt.figure(figsize=(10, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title(f'Class Distribution - {phase.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        dist_path = self.config['tensorboard']['log_dir'] / 'class_distributions'
        dist_path.mkdir(exist_ok=True)
        plt.savefig(dist_path / f'{phase}_epoch_{epoch}.png')
        plt.close()
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(
                f"{phase}/class_distribution",
                plt.gcf(),
                epoch
            )
            
    def log_sample_predictions(self, images, true_labels, pred_labels, class_names, phase, epoch):
        """Log sample predictions with images"""
        num_samples = min(8, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f'True: {class_names[true_labels[i]]}')
            axes[0, i].axis('off')
            
            # Prediction
            axes[1, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(f'Pred: {class_names[pred_labels[i]]}')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        
        # Save plot
        pred_path = self.config['tensorboard']['log_dir'] / 'sample_predictions'
        pred_path.mkdir(exist_ok=True)
        plt.savefig(pred_path / f'{phase}_epoch_{epoch}.png')
        plt.close()
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(
                f"{phase}/sample_predictions",
                plt.gcf(),
                epoch
            )
            
    def log_confidence_maps(self, images, confidences, phase, epoch):
        """Log confidence maps for predictions"""
        num_samples = min(8, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Confidence map
            conf_map = axes[1, i].imshow(confidences[i].cpu().numpy(), cmap='viridis')
            axes[1, i].set_title('Confidence')
            axes[1, i].axis('off')
            plt.colorbar(conf_map, ax=axes[1, i])
            
        plt.tight_layout()
        
        # Save plot
        conf_path = self.config['tensorboard']['log_dir'] / 'confidence_maps'
        conf_path.mkdir(exist_ok=True)
        plt.savefig(conf_path / f'{phase}_epoch_{epoch}.png')
        plt.close()
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(
                f"{phase}/confidence_maps",
                plt.gcf(),
                epoch
            )
            
    def log_model_architecture(self, model):
        """Log model architecture to TensorBoard"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_graph(model, torch.randn(1, 3, 224, 224))
            
    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters to TensorBoard"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hyperparams, {})
            
    def close(self):
        """Close TensorBoard writer and clear memory"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        self.memory_manager.clear_memory()

def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calculate various classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_metrics_to_csv(metrics_dict, filename):
    """Save metrics to CSV file"""
    df = pd.DataFrame(metrics_dict)
    df.to_csv(filename, index=False) 