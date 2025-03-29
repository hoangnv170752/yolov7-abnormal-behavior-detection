import os
import torch
import torch.nn as nn
import numpy as np


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """
    Load model and optimizer from checkpoint
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load checkpoint into
        
    Returns:
        epoch: Epoch of checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    epoch = 0
    if 'epoch' in checkpoint and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    
    return epoch


def save_checkpoint(model, filename, optimizer=None, epoch=None):
    """
    Save model and optimizer to checkpoint
    
    Args:
        model: Model to save
        filename: Path to save checkpoint to
        optimizer: Optional optimizer to save
        epoch: Optional epoch to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    state = {
        'model': model.state_dict(),
    }
    
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    if epoch is not None:
        state['epoch'] = epoch
    
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def init_weights(m):
    """
    Initialize model weights
    
    Args:
        m: Model module to initialize
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def calculate_metrics(outputs, targets):
    """
    Calculate classification metrics
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert logits to predictions
    if isinstance(outputs, dict) and 'behavior_logits' in outputs:
        outputs = outputs['behavior_logits']
    
    preds = torch.argmax(outputs, dim=1)
    
    # Calculate accuracy
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    
    # Calculate per-class metrics
    num_classes = outputs.size(1)
    precision = []
    recall = []
    f1 = []
    
    for c in range(num_classes):
        # True positives: predicted class c and actual class c
        tp = ((preds == c) & (targets == c)).sum().item()
        
        # False positives: predicted class c but actual class is not c
        fp = ((preds == c) & (targets != c)).sum().item()
        
        # False negatives: predicted class is not c but actual class is c
        fn = ((preds != c) & (targets == c)).sum().item()
        
        # Calculate precision, recall, and F1 score
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def combine_models(abnormal_model, ojr_model):
    """
    Combine abnormal behavior detection and OJR models into a single model
    
    Args:
        abnormal_model: Abnormal behavior detection model
        ojr_model: OJR model
        
    Returns:
        UnifiedModel: Combined model
    """
    from models.unified_model import UnifiedModel
    
    unified_model = UnifiedModel(abnormal_model, ojr_model)
    return unified_model
