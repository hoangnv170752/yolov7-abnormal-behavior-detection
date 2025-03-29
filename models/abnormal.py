import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AbnormalBehaviorDetector(nn.Module):
    """
    Abnormal Behavior Detection model that uses temporal features to classify behaviors
    
    Args:
        num_keypoints (int): Number of keypoints in the pose estimation
        hidden_dim (int): Hidden dimension of the model
        temporal_window (int): Number of frames to consider for temporal analysis
        num_classes (int): Number of behavior classes to detect
    """
    def __init__(self, num_keypoints=17, hidden_dim=192, temporal_window=5, num_classes=5):
        super(AbnormalBehaviorDetector, self).__init__()
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.num_classes = num_classes
        
        # Spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal feature extraction using LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism for focusing on important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, keypoints_seq, seq_lengths=None):
        """
        Forward pass through the abnormal behavior detector
        
        Args:
            keypoints_seq (torch.Tensor): Batch of keypoint sequences
                Shape: [batch_size, seq_len, num_keypoints*2]
            seq_lengths (torch.Tensor): Length of each sequence in the batch
                Shape: [batch_size]
                
        Returns:
            dict: Dictionary containing behavior probabilities and features
        """
        batch_size, seq_len, _ = keypoints_seq.shape
        
        # Process each frame with spatial encoder
        spatial_features = []
        for t in range(seq_len):
            keypoints = keypoints_seq[:, t, :]
            spatial_feat = self.spatial_encoder(keypoints)
            spatial_features.append(spatial_feat)
        
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Handle variable sequence lengths if provided
        if seq_lengths is not None:
            # Pack the sequence for efficient processing
            packed_features = pack_padded_sequence(
                spatial_features, 
                seq_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Process through LSTM
            packed_output, (hidden, _) = self.temporal_encoder(packed_features)
            
            # Unpack the sequence
            temporal_features, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Process through LSTM with fixed sequence length
            temporal_features, (hidden, _) = self.temporal_encoder(spatial_features)
        
        # Apply attention mechanism
        attention_scores = self.attention(temporal_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to get context vector
        context = torch.sum(attention_weights * temporal_features, dim=1)
        
        # Classify the behavior
        behavior_logits = self.classifier(context)
        behavior_probs = F.softmax(behavior_logits, dim=1)
        
        return {
            'behavior_logits': behavior_logits,
            'behavior_probs': behavior_probs,
            'features': context,
            'attention_weights': attention_weights
        }


class TemporalFeatureExtractor(nn.Module):
    """
    Extracts temporal features from a sequence of frames
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension of the model
    """
    def __init__(self, input_dim, hidden_dim):
        super(TemporalFeatureExtractor, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        """
        Forward pass through the temporal feature extractor
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Temporal features [batch_size, seq_len//2, hidden_dim*2]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape for 1D convolution [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply 1D convolution
        x = F.relu(self.bn(self.conv1d(x)))
        
        # Apply pooling to reduce sequence length
        x = self.pool(x)
        
        # Reshape back to [batch_size, seq_len//2, hidden_dim]
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        return x
