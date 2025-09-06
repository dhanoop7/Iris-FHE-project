import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any

def plot_sample(image: np.ndarray, 
               true_mask: np.ndarray, 
               pred_mask: np.ndarray,
               save_path: str = None):
    """Visualize segmentation results"""
    plt.figure(figsize=(15, 5))
    
    titles = ['Input Image', 'True Mask', 'Predicted Mask']
    images = [image, true_mask, pred_mask]
    
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def calculate_metrics(true_mask: np.ndarray, 
                     pred_mask: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    pred_bin = (pred_mask > threshold).astype(bool)
    true_bin = (true_mask > 0.5).astype(bool)
    
    intersection = np.logical_and(true_bin, pred_bin)
    union = np.logical_or(true_bin, pred_bin)
    
    metrics = {
        'iou': np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0,
        'precision': np.sum(intersection) / np.sum(pred_bin) if np.sum(pred_bin) > 0 else 0.0,
        'recall': np.sum(intersection) / np.sum(true_bin) if np.sum(true_bin) > 0 else 0.0
    }
    return metrics

def save_plots(history):
    """Save training curves"""
    plt.figure(figsize=(12, 4))
    metrics = ['loss', 'iou', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 4, i+1)
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'])
    
    plt.tight_layout()
    plt.savefig('output/training_metrics.png')
    plt.close()

def log_metrics(model, images, masks):
    """Save comprehensive evaluation"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation': {}
    }
    
    # Sample predictions
    sample_idx = np.random.choice(len(images), 5, replace=False)
    for idx in sample_idx:
        pred = model.predict(images[idx:idx+1][..., np.newaxis])[0]
        results['evaluation'][str(idx)] = calculate_metrics(masks[idx], pred)
    
    # Save to JSON
    with open('output/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)