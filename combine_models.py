#!/usr/bin/env python
"""
Script to combine trained abnormal behavior detection and OJR models into a single unified model.
This creates a combined model for inference that includes both abnormal behavior detection
and occluded joint recovery capabilities.
"""

import os
import torch
import argparse
import yaml
from pathlib import Path

from models.abnormal import AbnormalBehaviorDetector
from models.ojr import OJR
from models.unified_model import UnifiedModel
from models.model_utils import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine Abnormal Behavior Detection and OJR models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--abnormal-weights', type=str, help='Path to abnormal behavior detection weights (default from config)')
    parser.add_argument('--ojr-weights', type=str, help='Path to OJR weights (default from config)')
    parser.add_argument('--output', type=str, default='weights/unified_model.pt', help='Output path for unified model')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load abnormal behavior detection model
    print("Loading abnormal behavior detection model...")
    abnormal_model = AbnormalBehaviorDetector(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window'],
        num_classes=config['training']['behavior_classes']
    ).to(device)
    
    abnormal_weights = args.abnormal_weights or config['model']['save_path']
    if os.path.exists(abnormal_weights):
        load_checkpoint(abnormal_model, abnormal_weights)
        print(f"Loaded abnormal behavior detection weights from {abnormal_weights}")
    else:
        print(f"Warning: Abnormal behavior detection weights not found at {abnormal_weights}")
        return
    
    # Load OJR model
    print("Loading OJR model...")
    ojr_model = OJR(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    ).to(device)
    
    ojr_weights = args.ojr_weights or config['model']['ojr_save_path']
    if os.path.exists(ojr_weights):
        load_checkpoint(ojr_model, ojr_weights)
        print(f"Loaded OJR weights from {ojr_weights}")
    else:
        print(f"Warning: OJR weights not found at {ojr_weights}")
        return
    
    # Create unified model
    print("Creating unified model...")
    unified_model = UnifiedModel(abnormal_model, ojr_model)
    
    # Save unified model
    torch.save({
        'model': unified_model.state_dict(),
        'config': config,
        'abnormal_path': abnormal_weights,
        'ojr_path': ojr_weights
    }, args.output)
    
    print(f"Unified model saved to {args.output}")
    print("\nTo use the unified model for inference:")
    print(f"python detect_abnormal.py --weights {args.output} --unified")


if __name__ == "__main__":
    main()
