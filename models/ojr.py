import torch
import torch.nn as nn
import torch.nn.functional as F


class OJR(nn.Module):
    """
    Occluded Joint Recovery (OJR) model that recovers occluded keypoints
    
    Args:
        num_keypoints (int): Number of keypoints in the pose estimation
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of layers in the model
    """
    def __init__(self, num_keypoints=17, hidden_dim=96, num_layers=2):
        super(OJR, self).__init__()
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layers
        self.input_fc = nn.Linear(num_keypoints * 2, hidden_dim)
        
        # Visibility estimation
        self.visibility_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_keypoints),
            nn.Sigmoid()
        )
        
        # Joint recovery layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_dim + num_keypoints, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.recovery_layers = nn.Sequential(*layers)
        self.output_fc = nn.Linear(hidden_dim, num_keypoints * 2)
        
    def forward(self, keypoints, occlusion_mask=None):
        """
        Forward pass through the OJR model
        
        Args:
            keypoints (torch.Tensor): Batch of keypoints [batch_size, num_keypoints*2]
            occlusion_mask (torch.Tensor, optional): Mask indicating occluded keypoints
                Shape: [batch_size, num_keypoints]
                
        Returns:
            dict: Dictionary containing recovered keypoints and visibility scores
        """
        batch_size = keypoints.shape[0]
        
        # Extract features from input keypoints
        features = F.relu(self.input_fc(keypoints))
        
        # Estimate visibility if not provided
        if occlusion_mask is None:
            visibility_scores = self.visibility_estimator(features)
            occlusion_mask = visibility_scores < 0.5
        else:
            visibility_scores = 1.0 - occlusion_mask.float()
        
        # Concatenate features with occlusion mask
        recovery_input = torch.cat([features, occlusion_mask.float()], dim=1)
        
        # Process through recovery layers
        recovery_features = self.recovery_layers(recovery_input)
        
        # Generate recovered keypoints
        recovered_keypoints_delta = self.output_fc(recovery_features)
        
        # Only apply recovery to occluded keypoints
        recovered_keypoints = keypoints.clone()
        
        # Reshape for easier indexing
        keypoints_reshaped = keypoints.view(batch_size, self.num_keypoints, 2)
        recovered_delta_reshaped = recovered_keypoints_delta.view(batch_size, self.num_keypoints, 2)
        
        # Create a mask for indexing
        mask_expanded = occlusion_mask.unsqueeze(-1).expand(-1, -1, 2)
        mask_flat = mask_expanded.reshape(batch_size, -1)
        
        # Apply the recovered keypoints only to occluded joints
        recovered_keypoints[mask_flat] = (keypoints + recovered_keypoints_delta)[mask_flat]
        
        return {
            'recovered_keypoints': recovered_keypoints,
            'visibility_scores': visibility_scores,
            'recovery_delta': recovered_keypoints_delta
        }


class SpatialTemporalOJR(nn.Module):
    """
    Spatial-Temporal Occluded Joint Recovery model that uses both spatial and temporal information
    
    Args:
        num_keypoints (int): Number of keypoints in the pose estimation
        hidden_dim (int): Hidden dimension of the model
        temporal_window (int): Number of frames to consider for temporal recovery
    """
    def __init__(self, num_keypoints=17, hidden_dim=96, temporal_window=5):
        super(SpatialTemporalOJR, self).__init__()
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        
        # Spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal feature extraction
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Recovery decoder
        self.recovery_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_keypoints, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_keypoints * 2)
        )
        
    def forward(self, keypoints_seq, occlusion_mask):
        """
        Forward pass through the spatial-temporal OJR model
        
        Args:
            keypoints_seq (torch.Tensor): Sequence of keypoints
                Shape: [batch_size, seq_len, num_keypoints*2]
            occlusion_mask (torch.Tensor): Mask indicating occluded keypoints
                Shape: [batch_size, num_keypoints]
                
        Returns:
            dict: Dictionary containing recovered keypoints
        """
        batch_size, seq_len, _ = keypoints_seq.shape
        
        # Extract spatial features for each frame
        spatial_features = []
        for t in range(seq_len):
            keypoints = keypoints_seq[:, t, :]
            spatial_feat = self.spatial_encoder(keypoints)
            spatial_features.append(spatial_feat)
        
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Extract temporal features
        temporal_features, _ = self.temporal_encoder(spatial_features)
        
        # Get features for the middle frame (current frame)
        mid_idx = seq_len // 2
        current_features = temporal_features[:, mid_idx, :]
        
        # Concatenate with occlusion mask
        recovery_input = torch.cat([current_features, occlusion_mask.float()], dim=1)
        
        # Generate recovered keypoints
        recovered_keypoints_delta = self.recovery_decoder(recovery_input)
        
        # Get current frame keypoints
        current_keypoints = keypoints_seq[:, mid_idx, :]
        
        # Only apply recovery to occluded keypoints
        recovered_keypoints = current_keypoints.clone()
        
        # Reshape for easier indexing
        keypoints_reshaped = current_keypoints.view(batch_size, self.num_keypoints, 2)
        recovered_delta_reshaped = recovered_keypoints_delta.view(batch_size, self.num_keypoints, 2)
        
        # Create a mask for indexing
        mask_expanded = occlusion_mask.unsqueeze(-1).expand(-1, -1, 2)
        mask_flat = mask_expanded.reshape(batch_size, -1)
        
        # Apply the recovered keypoints only to occluded joints
        recovered_keypoints[mask_flat] = (current_keypoints + recovered_keypoints_delta)[mask_flat]
        
        return {
            'recovered_keypoints': recovered_keypoints,
            'recovery_delta': recovered_keypoints_delta
        }
