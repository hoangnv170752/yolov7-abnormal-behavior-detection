#!/usr/bin/env python
"""
Training script for abnormal behavior detection and OJR models
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.abnormal import AbnormalBehaviorDetector
from models.ojr import OJR
from models.model_utils import save_checkpoint, load_checkpoint, calculate_metrics
from utils.data_utils import AbnormalDataset, OJRDataset, custom_collate_fn
from utils.losses import FocalLoss


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--train-abnormal', action='store_true', help='Train abnormal behavior detection model')
    parser.add_argument('--train-ojr', action='store_true', help='Train OJR model')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use')
    
    return parser.parse_args()


def train_abnormal(config, args):
    """
    Train the abnormal behavior detection model
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
    
    # Create datasets
    train_dataset = AbnormalDataset(
        list_file=config['data']['train_list'],
        img_size=config['data']['img_size'],
        temporal_window=config['model']['temporal_window'],
        is_train=True
    )
    
    val_dataset = AbnormalDataset(
        list_file=config['data']['val_list'],
        img_size=config['data']['img_size'],
        temporal_window=config['model']['temporal_window'],
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = AbnormalBehaviorDetector(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window'],
        num_classes=config['training']['behavior_classes']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    # Use class weights to handle imbalanced data
    class_weights = torch.ones(config['training']['behavior_classes'], device=device)
    # Increase weight for abnormal classes (1-4)
    for i in range(1, config['training']['behavior_classes']):
        class_weights[i] = config['training']['abnormal_class_weight']
    
    # Use focal loss for better handling of imbalanced data
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0,
        reduction='mean'
    )
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, config['model']['save_path'], optimizer)
    
    # Training loop
    num_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    print(f"Starting training from epoch {start_epoch + 1}")
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            keypoints_seq, labels, seq_lengths = batch
            
            # Move to device
            keypoints_seq = keypoints_seq.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)
            
            # Forward pass
            outputs = model(keypoints_seq, seq_lengths)
            
            # Calculate loss
            loss = criterion(outputs['behavior_logits'], labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            batch_metrics = calculate_metrics(outputs['behavior_logits'], labels)
            
            for k in train_metrics.keys():
                train_metrics[k] += batch_metrics[k]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': batch_metrics['accuracy']
            })
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        for k in train_metrics.keys():
            train_metrics[k] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for batch_idx, batch in enumerate(pbar):
                keypoints_seq, labels, seq_lengths = batch
                
                # Move to device
                keypoints_seq = keypoints_seq.to(device)
                labels = labels.to(device)
                seq_lengths = seq_lengths.to(device)
                
                # Forward pass
                outputs = model(keypoints_seq, seq_lengths)
                
                # Calculate loss
                loss = criterion(outputs['behavior_logits'], labels)
                
                # Update metrics
                val_loss += loss.item()
                batch_metrics = calculate_metrics(outputs['behavior_logits'], labels)
                
                for k in val_metrics.keys():
                    val_metrics[k] += batch_metrics[k]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': batch_metrics['accuracy']
                })
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        for k in val_metrics.keys():
            val_metrics[k] /= len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['macro_f1']:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            save_checkpoint(model, config['model']['save_path'], optimizer, epoch + 1)
        
        # Save checkpoint if F1 score improved
        if val_metrics['macro_f1'] > best_f1:
            print(f"Validation F1 improved from {best_f1:.4f} to {val_metrics['macro_f1']:.4f}")
            best_f1 = val_metrics['macro_f1']
            save_checkpoint(
                model,
                config['model']['save_path'].replace('.pt', '_best_f1.pt'),
                optimizer,
                epoch + 1
            )
    
    print("Training completed!")


def train_ojr(config, args):
    """
    Train the OJR model
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model']['ojr_save_path']), exist_ok=True)
    
    # Create datasets
    train_dataset = OJRDataset(
        list_file=config['data']['train_list'],
        img_size=config['data']['img_size'],
        occlusion_threshold=config['training']['occlusion_threshold'],
        is_train=True
    )
    
    val_dataset = OJRDataset(
        list_file=config['data']['val_list'],
        img_size=config['data']['img_size'],
        occlusion_threshold=config['training']['occlusion_threshold'],
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = OJR(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['ojr_lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    criterion = nn.MSELoss(reduction='none')
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, config['model']['ojr_save_path'], optimizer)
    
    # Training loop
    num_epochs = config['training']['ojr_epochs']
    best_val_loss = float('inf')
    
    print(f"Starting training from epoch {start_epoch + 1}")
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            keypoints, occluded_keypoints, occlusion_mask = batch
            
            # Move to device
            keypoints = keypoints.to(device)
            occluded_keypoints = occluded_keypoints.to(device)
            occlusion_mask = occlusion_mask.to(device)
            
            # Forward pass
            outputs = model(keypoints, occlusion_mask)
            
            # Calculate loss (only for occluded keypoints)
            loss_mask = occlusion_mask.unsqueeze(-1).repeat(1, 1, 2).reshape(keypoints.shape)
            
            # MSE loss between recovered and ground truth keypoints
            mse_loss = criterion(outputs['recovered_keypoints'], occluded_keypoints)
            
            # Apply mask to only consider occluded keypoints
            masked_loss = (mse_loss * loss_mask.float()).sum() / (loss_mask.sum() + 1e-8)
            
            # Backward pass
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += masked_loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': masked_loss.item()})
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for batch_idx, batch in enumerate(pbar):
                keypoints, occluded_keypoints, occlusion_mask = batch
                
                # Move to device
                keypoints = keypoints.to(device)
                occluded_keypoints = occluded_keypoints.to(device)
                occlusion_mask = occlusion_mask.to(device)
                
                # Forward pass
                outputs = model(keypoints, occlusion_mask)
                
                # Calculate loss (only for occluded keypoints)
                loss_mask = occlusion_mask.unsqueeze(-1).repeat(1, 1, 2).reshape(keypoints.shape)
                
                # MSE loss between recovered and ground truth keypoints
                mse_loss = criterion(outputs['recovered_keypoints'], occluded_keypoints)
                
                # Apply mask to only consider occluded keypoints
                masked_loss = (mse_loss * loss_mask.float()).sum() / (loss_mask.sum() + 1e-8)
                
                # Update metrics
                val_loss += masked_loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': masked_loss.item()})
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            save_checkpoint(model, config['model']['ojr_save_path'], optimizer, epoch + 1)
    
    print("Training completed!")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train models
    if args.train_abnormal:
        print("Training abnormal behavior detection model...")
        train_abnormal(config, args)
    
    if args.train_ojr:
        print("Training OJR model...")
        train_ojr(config, args)


if __name__ == "__main__":
    main()
