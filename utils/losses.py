import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    
    Focal Loss was introduced in the paper "Focal Loss for Dense Object Detection"
    by Lin et al. (https://arxiv.org/abs/1708.02002). It helps to address class
    imbalance by down-weighting easy examples and focusing on hard examples.
    
    Args:
        alpha (torch.Tensor, optional): Weight for each class. Must be a tensor
            of size C. Default: None (all classes have weight 1)
        gamma (float, optional): Focusing parameter. Default: 2.0
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is the batch size
                and C is the number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N,)
                
        Returns:
            torch.Tensor: Computed loss
        """
        log_softmax = F.log_softmax(inputs, dim=1)
        
        # Gather the log-probabilities of the target classes
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)
        pt = torch.exp(log_pt)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            log_pt = log_pt * alpha_t
        
        # Calculate focal loss
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        
        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class SmoothL1KeypointLoss(nn.Module):
    """
    Smooth L1 loss for keypoint regression with visibility weighting.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        beta (float, optional): Threshold parameter for Smooth L1 loss. Default: 1.0
    """
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1KeypointLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, pred_keypoints, target_keypoints, visibility=None):
        """
        Forward pass of the Smooth L1 Keypoint Loss.
        
        Args:
            pred_keypoints (torch.Tensor): Predicted keypoints of shape (N, K*2) or (N, K, 2)
                where N is the batch size and K is the number of keypoints
            target_keypoints (torch.Tensor): Ground truth keypoints of shape (N, K*2) or (N, K, 2)
            visibility (torch.Tensor, optional): Visibility mask for keypoints of shape (N, K)
                where 1 indicates visible keypoint and 0 indicates invisible keypoint
                
        Returns:
            torch.Tensor: Computed loss
        """
        # Ensure keypoints are in the same shape
        if pred_keypoints.dim() == 3:
            pred_flat = pred_keypoints.reshape(pred_keypoints.size(0), -1)
        else:
            pred_flat = pred_keypoints
            
        if target_keypoints.dim() == 3:
            target_flat = target_keypoints.reshape(target_keypoints.size(0), -1)
        else:
            target_flat = target_keypoints
        
        # Calculate Smooth L1 loss
        diff = torch.abs(pred_flat - target_flat)
        loss = torch.where(diff < self.beta, 
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        # Apply visibility mask if provided
        if visibility is not None:
            # Expand visibility mask to match keypoint dimensions
            if visibility.dim() == 2:
                # Visibility is (N, K), expand to (N, K*2)
                vis_expanded = visibility.unsqueeze(-1).repeat(1, 1, 2).reshape(visibility.size(0), -1)
            else:
                vis_expanded = visibility
                
            # Apply mask
            loss = loss * vis_expanded
            
            # Normalize by the number of visible keypoints
            num_visible = vis_expanded.sum().clamp(min=1.0)
            
            if self.reduction == 'mean':
                return loss.sum() / num_visible
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining behavior classification and keypoint regression.
    
    Args:
        behavior_loss_weight (float): Weight for behavior classification loss
        keypoint_loss_weight (float): Weight for keypoint regression loss
        behavior_loss (nn.Module): Loss function for behavior classification
        keypoint_loss (nn.Module): Loss function for keypoint regression
    """
    def __init__(self, behavior_loss_weight=0.5, keypoint_loss_weight=0.5,
                 behavior_loss=None, keypoint_loss=None):
        super(MultiTaskLoss, self).__init__()
        self.behavior_loss_weight = behavior_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        
        # Default loss functions if not provided
        self.behavior_loss = behavior_loss or FocalLoss(gamma=2.0)
        self.keypoint_loss = keypoint_loss or SmoothL1KeypointLoss()
        
    def forward(self, outputs, targets):
        """
        Forward pass of the Multi-task Loss.
        
        Args:
            outputs (dict): Dictionary containing model outputs
                - behavior_logits: Predicted behavior logits
                - keypoints: Predicted keypoints
            targets (dict): Dictionary containing ground truth
                - behavior: Ground truth behavior labels
                - keypoints: Ground truth keypoints
                - visibility: Optional visibility mask for keypoints
                
        Returns:
            torch.Tensor: Computed total loss
            dict: Dictionary containing individual losses
        """
        losses = {}
        
        # Behavior classification loss
        if 'behavior_logits' in outputs and 'behavior' in targets:
            losses['behavior_loss'] = self.behavior_loss(
                outputs['behavior_logits'], 
                targets['behavior']
            )
        
        # Keypoint regression loss
        if 'keypoints' in outputs and 'keypoints' in targets:
            visibility = targets.get('visibility', None)
            losses['keypoint_loss'] = self.keypoint_loss(
                outputs['keypoints'],
                targets['keypoints'],
                visibility
            )
        
        # Calculate total loss
        total_loss = 0.0
        if 'behavior_loss' in losses:
            total_loss += self.behavior_loss_weight * losses['behavior_loss']
        if 'keypoint_loss' in losses:
            total_loss += self.keypoint_loss_weight * losses['keypoint_loss']
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses
