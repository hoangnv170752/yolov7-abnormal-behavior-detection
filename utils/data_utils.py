import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import Path


class AbnormalDataset(Dataset):
    """
    Dataset for abnormal behavior detection
    
    Args:
        list_file (str): Path to text file containing video paths and labels
        img_size (int): Size of input images
        temporal_window (int): Number of frames to sample from each video
        is_train (bool): Whether this is for training or evaluation
    """
    def __init__(self, list_file, img_size=416, temporal_window=5, is_train=True):
        self.list_file = list_file
        self.img_size = img_size
        self.temporal_window = temporal_window
        self.is_train = is_train
        
        # Load video paths and labels
        self.samples = self._load_samples()
        
        # Augmentation for training
        self.augmenter = VideoAugmenter() if is_train else None
        
    def _load_samples(self):
        """Load video paths and labels from list file"""
        samples = []
        
        try:
            with open(self.list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        print(f"Warning: Invalid line in {self.list_file}: {line}")
                        continue
                    
                    video_path = parts[0]
                    label = int(parts[1])
                    
                    samples.append((video_path, label))
        except Exception as e:
            print(f"Error loading samples from {self.list_file}: {str(e)}")
            return []
        
        print(f"Loaded {len(samples)} samples from {self.list_file}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a video clip and its label"""
        video_path, label = self.samples[idx]
        
        try:
            # Load video frames
            frames, keypoints = self._load_video(video_path)
            
            # Convert to tensor
            frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0
            
            # Apply augmentation if training
            if self.is_train and self.augmenter is not None:
                frames_tensor, keypoints = self.augmenter(frames_tensor, keypoints)
            
            # Create sequence length
            seq_length = frames_tensor.shape[0]
            
            return frames_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(seq_length, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a dummy sample
            dummy_frames = torch.zeros((self.temporal_window, 3, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_frames, torch.tensor(0, dtype=torch.long), torch.tensor(self.temporal_window, dtype=torch.long)
    
    def _load_video(self, video_path):
        """Load video frames and extract keypoints"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Sample frames
        if frame_count <= self.temporal_window:
            # If video is shorter than temporal window, duplicate frames
            frame_indices = list(range(frame_count))
            frame_indices = frame_indices + [frame_indices[-1]] * (self.temporal_window - frame_count)
        else:
            # Randomly sample frames
            if self.is_train:
                start_idx = random.randint(0, frame_count - self.temporal_window)
            else:
                start_idx = (frame_count - self.temporal_window) // 2  # Center frames for validation
            
            frame_indices = list(range(start_idx, start_idx + self.temporal_window))
        
        # Load frames
        frames = []
        keypoints = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Could not read frame {idx} from video: {video_path}")
            
            # Resize frame
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transpose for PyTorch: (H, W, C) -> (C, H, W)
            frame = frame.transpose(2, 0, 1)
            
            frames.append(frame)
            
            # TODO: Extract keypoints using YOLOv7 pose estimation
            # For now, use dummy keypoints
            dummy_keypoints = np.zeros((17, 2))
            keypoints.append(dummy_keypoints)
        
        cap.release()
        
        # Stack frames
        frames = np.stack(frames, axis=0)
        keypoints = np.stack(keypoints, axis=0)
        
        return frames, keypoints


class OJRDataset(Dataset):
    """
    Dataset for Occluded Joint Recovery (OJR)
    
    Args:
        list_file (str): Path to text file containing video paths and labels
        img_size (int): Size of input images
        occlusion_threshold (float): Threshold for considering a keypoint as occluded
        is_train (bool): Whether this is for training or evaluation
    """
    def __init__(self, list_file, img_size=416, occlusion_threshold=0.3, is_train=True):
        self.list_file = list_file
        self.img_size = img_size
        self.occlusion_threshold = occlusion_threshold
        self.is_train = is_train
        
        # Load video paths and labels
        self.samples = self._load_samples()
        
        # Augmentation for training
        self.augmenter = KeypointAugmenter() if is_train else None
        
    def _load_samples(self):
        """Load video paths and labels from list file"""
        samples = []
        
        try:
            with open(self.list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        print(f"Warning: Invalid line in {self.list_file}: {line}")
                        continue
                    
                    video_path = parts[0]
                    label = int(parts[1])
                    
                    samples.append((video_path, label))
        except Exception as e:
            print(f"Error loading samples from {self.list_file}: {str(e)}")
            return []
        
        print(f"Loaded {len(samples)} samples from {self.list_file}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get keypoints and occlusion mask"""
        video_path, _ = self.samples[idx]
        
        try:
            # Load keypoints
            keypoints = self._load_keypoints(video_path)
            
            # Convert to tensor
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
            
            # Create artificial occlusions for training
            if self.is_train:
                occluded_keypoints, occlusion_mask = self._create_occlusions(keypoints_tensor)
            else:
                # For validation, use a fixed occlusion pattern
                occluded_keypoints, occlusion_mask = self._create_fixed_occlusions(keypoints_tensor)
            
            # Apply augmentation if training
            if self.is_train and self.augmenter is not None:
                keypoints_tensor, occluded_keypoints = self.augmenter(keypoints_tensor, occluded_keypoints)
            
            # Flatten keypoints: (17, 2) -> (34,)
            keypoints_flat = keypoints_tensor.reshape(-1)
            occluded_keypoints_flat = occluded_keypoints.reshape(-1)
            
            return keypoints_flat, occluded_keypoints_flat, occlusion_mask
        
        except Exception as e:
            print(f"Error loading keypoints from {video_path}: {str(e)}")
            # Return a dummy sample
            dummy_keypoints = torch.zeros(17 * 2, dtype=torch.float32)
            dummy_occluded = torch.zeros(17 * 2, dtype=torch.float32)
            dummy_mask = torch.zeros(17, dtype=torch.bool)
            dummy_mask[0] = True  # At least one occluded keypoint
            
            return dummy_keypoints, dummy_occluded, dummy_mask
    
    def _load_keypoints(self, video_path):
        """Load keypoints from video or keypoint file"""
        # TODO: Implement actual keypoint loading
        # For now, use dummy keypoints
        keypoints = np.random.rand(17, 2) * self.img_size
        
        return keypoints
    
    def _create_occlusions(self, keypoints):
        """Create artificial occlusions for training"""
        num_keypoints = keypoints.shape[0]
        
        # Create a copy of the original keypoints
        occluded_keypoints = keypoints.clone()
        
        # Randomly select keypoints to occlude
        num_to_occlude = random.randint(1, num_keypoints // 2)
        occluded_indices = random.sample(range(num_keypoints), num_to_occlude)
        
        # Create occlusion mask
        occlusion_mask = torch.zeros(num_keypoints, dtype=torch.bool)
        occlusion_mask[occluded_indices] = True
        
        # Modify occluded keypoints
        for idx in occluded_indices:
            # Add noise to occluded keypoints
            noise = (torch.rand(2) - 0.5) * self.img_size * 0.2
            occluded_keypoints[idx] = keypoints[idx] + noise
        
        return occluded_keypoints, occlusion_mask
    
    def _create_fixed_occlusions(self, keypoints):
        """Create fixed occlusions for validation"""
        num_keypoints = keypoints.shape[0]
        
        # Create a copy of the original keypoints
        occluded_keypoints = keypoints.clone()
        
        # Occlude specific keypoints (e.g., ankles, wrists)
        occluded_indices = [9, 10, 15, 16]  # Assuming COCO keypoint format
        
        # Create occlusion mask
        occlusion_mask = torch.zeros(num_keypoints, dtype=torch.bool)
        for idx in occluded_indices:
            if idx < num_keypoints:
                occlusion_mask[idx] = True
        
        # Modify occluded keypoints
        for idx in occluded_indices:
            if idx < num_keypoints:
                # Add fixed noise to occluded keypoints
                noise = torch.tensor([self.img_size * 0.1, self.img_size * 0.1])
                occluded_keypoints[idx] = keypoints[idx] + noise
        
        return occluded_keypoints, occlusion_mask


class VideoAugmenter:
    """
    Augmentation for video frames and keypoints
    """
    def __init__(self, flip_prob=0.5, rotate_prob=0.3, max_rotation=15):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotation = max_rotation
    
    def __call__(self, frames, keypoints=None):
        """
        Apply augmentation to frames and keypoints
        
        Args:
            frames (torch.Tensor): Video frames [T, C, H, W]
            keypoints (np.ndarray, optional): Keypoints [T, K, 2]
            
        Returns:
            torch.Tensor: Augmented frames
            np.ndarray: Augmented keypoints (if provided)
        """
        # Ensure frames is a tensor
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        T, C, H, W = frames.shape
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            frames = torch.flip(frames, dims=[3])
            
            if keypoints is not None:
                # Flip x-coordinates
                keypoints[..., 0] = W - keypoints[..., 0]
        
        # Color jitter
        frames = self._color_jitter(frames)
        
        return frames, keypoints
    
    def _color_jitter(self, frames):
        """Apply color jittering to frames"""
        # Random brightness
        brightness_factor = random.uniform(0.8, 1.2)
        frames = frames * brightness_factor
        
        # Random contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = frames.mean(dim=[1, 2, 3], keepdim=True)
        frames = (frames - mean) * contrast_factor + mean
        
        # Clip values to [0, 1]
        frames = torch.clamp(frames, 0, 1)
        
        return frames


class KeypointAugmenter:
    """
    Augmentation for keypoints
    """
    def __init__(self, flip_prob=0.5, scale_prob=0.5, scale_range=(0.8, 1.2)):
        self.flip_prob = flip_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range
    
    def __call__(self, keypoints, occluded_keypoints):
        """
        Apply augmentation to keypoints
        
        Args:
            keypoints (torch.Tensor): Original keypoints [K, 2]
            occluded_keypoints (torch.Tensor): Occluded keypoints [K, 2]
            
        Returns:
            torch.Tensor: Augmented keypoints
            torch.Tensor: Augmented occluded keypoints
        """
        K, _ = keypoints.shape
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            # Assume keypoints are normalized to [0, 1]
            keypoints[:, 0] = 1 - keypoints[:, 0]
            occluded_keypoints[:, 0] = 1 - occluded_keypoints[:, 0]
            
            # Swap left-right keypoints (e.g., shoulders, hips)
            # Assuming COCO keypoint format
            pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for left, right in pairs:
                if left < K and right < K:
                    keypoints[[left, right]] = keypoints[[right, left]]
                    occluded_keypoints[[left, right]] = occluded_keypoints[[right, left]]
        
        # Scale
        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)
            center = torch.mean(keypoints, dim=0)
            
            keypoints = center + (keypoints - center) * scale
            occluded_keypoints = center + (occluded_keypoints - center) * scale
        
        return keypoints, occluded_keypoints


class OverSampler:
    """
    Oversample minority class samples
    
    Args:
        dataset: Dataset to oversample
        class_counts (dict): Number of samples per class
        target_ratio (float): Target ratio of minority to majority class
    """
    def __init__(self, dataset, class_counts, target_ratio=0.5):
        self.dataset = dataset
        self.class_counts = class_counts
        self.target_ratio = target_ratio
        
        # Find majority and minority classes
        self.majority_class = max(class_counts, key=class_counts.get)
        self.majority_count = class_counts[self.majority_class]
        
        # Calculate target counts for each class
        self.target_counts = {}
        for cls, count in class_counts.items():
            if cls == self.majority_class:
                self.target_counts[cls] = count
            else:
                self.target_counts[cls] = max(count, int(self.majority_count * target_ratio))
        
        # Create indices for each class
        self.class_indices = self._get_class_indices()
        
        # Create oversampled indices
        self.indices = self._create_indices()
    
    def _get_class_indices(self):
        """Get indices for each class"""
        class_indices = {}
        
        for cls in self.class_counts.keys():
            class_indices[cls] = []
        
        for i in range(len(self.dataset)):
            try:
                _, label, _ = self.dataset[i]
                cls = label.item()
                class_indices.setdefault(cls, []).append(i)
            except Exception as e:
                print(f"Error getting class for index {i}: {str(e)}")
        
        return class_indices
    
    def _create_indices(self):
        """Create oversampled indices"""
        indices = []
        
        for cls, target_count in self.target_counts.items():
            cls_indices = self.class_indices.get(cls, [])
            
            if not cls_indices:
                continue
            
            # If we need more samples than we have, oversample with replacement
            if target_count > len(cls_indices):
                # Original samples
                indices.extend(cls_indices)
                
                # Oversampled samples
                additional_needed = target_count - len(cls_indices)
                oversampled = random.choices(cls_indices, k=additional_needed)
                indices.extend(oversampled)
            else:
                # If we have enough samples, just take a random subset
                sampled = random.sample(cls_indices, target_count)
                indices.extend(sampled)
        
        # Shuffle indices
        random.shuffle(indices)
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def custom_collate_fn(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: List of tuples (frames, label, seq_length)
        
    Returns:
        torch.Tensor: Padded frames
        torch.Tensor: Labels
        torch.Tensor: Sequence lengths
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    
    if not batch:
        # Return empty tensors if batch is empty
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    frames, labels, seq_lengths = zip(*batch)
    
    # Get max sequence length
    max_seq_len = max(seq_lengths)
    
    # Get frame dimensions
    _, C, H, W = frames[0].shape
    
    # Create padded frames tensor
    padded_frames = torch.zeros((len(batch), max_seq_len, C, H, W), dtype=frames[0].dtype)
    
    # Fill padded frames tensor
    for i, (frame, seq_len) in enumerate(zip(frames, seq_lengths)):
        padded_frames[i, :seq_len] = frame
    
    # Stack labels and sequence lengths
    labels = torch.stack(labels)
    seq_lengths = torch.stack(seq_lengths)
    
    return padded_frames, labels, seq_lengths
