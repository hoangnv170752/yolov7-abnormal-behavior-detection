import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedModel(nn.Module):
    """
    Unified model that combines abnormal behavior detection and OJR capabilities.
    
    Args:
        abnormal_model: Pre-trained abnormal behavior detection model
        ojr_model: Pre-trained OJR model
    """
    def __init__(self, abnormal_model, ojr_model):
        super(UnifiedModel, self).__init__()
        self.abnormal_model = abnormal_model
        self.ojr_model = ojr_model
        
    def forward(self, frames, keypoints=None, occlusion_mask=None):
        """
        Forward pass through the unified model.
        
        Args:
            frames: Input video frames [batch_size, num_frames, channels, height, width]
            keypoints: Optional keypoints [batch_size, num_keypoints*2]
            occlusion_mask: Optional occlusion mask [batch_size, num_keypoints]
            
        Returns:
            Dictionary with abnormal behavior detection and joint recovery outputs
        """
        batch_size = frames.shape[0]
        
        # Process through abnormal behavior detection model
        abnormal_outputs = self.abnormal_model(frames)
        
        # If keypoints are not provided, use the ones from abnormal detection model
        if keypoints is None:
            keypoints = abnormal_outputs.get('keypoints', None)
        
        # Process through OJR model if keypoints are available and occlusion is detected
        if keypoints is not None:
            if occlusion_mask is None:
                # Estimate occlusion mask from keypoints
                visibility = abnormal_outputs.get('visibility', None)
                if visibility is not None:
                    occlusion_mask = visibility < 0.5
                else:
                    # Default occlusion mask (no occlusion)
                    occlusion_mask = torch.zeros(
                        (batch_size, self.ojr_model.num_keypoints), 
                        device=keypoints.device
                    ).bool()
            
            # Only process through OJR if there are occluded keypoints
            if occlusion_mask.sum() > 0:
                ojr_outputs = self.ojr_model(keypoints, occlusion_mask)
                recovered_keypoints = ojr_outputs['recovered_keypoints']
            else:
                recovered_keypoints = keypoints
                ojr_outputs = {'recovered_keypoints': keypoints}
        else:
            # No keypoints available
            recovered_keypoints = None
            ojr_outputs = {}
            
        # Combine outputs
        outputs = {
            'behavior_logits': abnormal_outputs.get('behavior_logits', None),
            'behavior_probs': abnormal_outputs.get('behavior_probs', None),
            'keypoints': keypoints,
            'recovered_keypoints': recovered_keypoints,
            'features': abnormal_outputs.get('features', None),
            'attention_weights': abnormal_outputs.get('attention_weights', None)
        }
        
        return outputs


class EndToEndUnifiedModel(nn.Module):
    """
    End-to-end unified model that includes YOLOv7 backbone, abnormal behavior detection,
    and OJR capabilities.
    
    Args:
        yolo_model: YOLOv7 model for person detection and keypoint estimation
        abnormal_model: Abnormal behavior detection model
        ojr_model: OJR model
        num_keypoints: Number of keypoints in the pose estimation
    """
    def __init__(self, yolo_model, abnormal_model, ojr_model, num_keypoints=17):
        super(EndToEndUnifiedModel, self).__init__()
        self.yolo_model = yolo_model
        self.abnormal_model = abnormal_model
        self.ojr_model = ojr_model
        self.num_keypoints = num_keypoints
        
    def forward(self, frames):
        """
        Forward pass through the end-to-end unified model.
        
        Args:
            frames: Input video frames [batch_size, num_frames, channels, height, width]
            
        Returns:
            Dictionary with detection, abnormal behavior detection, and joint recovery outputs
        """
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Process each frame with YOLOv7
        detections = []
        keypoints_seq = []
        
        for t in range(seq_len):
            frame = frames[:, t, :, :, :]
            
            # Get detections from YOLOv7
            yolo_output = self.yolo_model(frame)
            
            # Extract person detections and keypoints
            person_dets = []
            frame_keypoints = []
            
            for i in range(batch_size):
                # Filter person detections
                persons = [det for det in yolo_output[i] if det[5] == 0]  # Assuming 0 is person class
                
                if persons:
                    # Use the highest confidence person detection
                    best_person = max(persons, key=lambda x: x[4])
                    person_dets.append(best_person[:5])  # box + conf
                    
                    # Extract keypoints if available
                    if len(best_person) > 6:
                        kpts = best_person[6:].reshape(-1, 3)[:, :2].flatten()  # x,y coords only
                        frame_keypoints.append(kpts)
                    else:
                        # No keypoints, use zeros
                        frame_keypoints.append(torch.zeros(self.num_keypoints * 2))
                else:
                    # No person detected
                    person_dets.append(torch.zeros(5))
                    frame_keypoints.append(torch.zeros(self.num_keypoints * 2))
            
            detections.append(torch.stack(person_dets))
            keypoints_seq.append(torch.stack(frame_keypoints))
        
        detections = torch.stack(detections, dim=1)
        keypoints_seq = torch.stack(keypoints_seq, dim=1)
        
        # Process through abnormal behavior detector
        abnormal_outputs = self.abnormal_model(keypoints_seq)
        
        # Get current frame keypoints (middle frame)
        mid_idx = seq_len // 2
        current_keypoints = keypoints_seq[:, mid_idx, :]
        
        # Estimate occlusion from keypoints
        visibility = torch.norm(current_keypoints.reshape(batch_size, self.num_keypoints, 2), dim=2) > 0
        occlusion_mask = ~visibility
        
        # Process through OJR if there are occluded keypoints
        if occlusion_mask.sum() > 0:
            ojr_outputs = self.ojr_model(current_keypoints, occlusion_mask)
            recovered_keypoints = ojr_outputs['recovered_keypoints']
        else:
            recovered_keypoints = current_keypoints
            ojr_outputs = {'recovered_keypoints': current_keypoints}
            
        # Combine outputs
        outputs = {
            'detections': detections,
            'behavior_logits': abnormal_outputs.get('behavior_logits', None),
            'behavior_probs': abnormal_outputs.get('behavior_probs', None),
            'keypoints': current_keypoints,
            'keypoints_seq': keypoints_seq,
            'recovered_keypoints': recovered_keypoints,
            'occlusion_mask': occlusion_mask,
            'features': abnormal_outputs.get('features', None),
            'attention_weights': abnormal_outputs.get('attention_weights', None)
        }
        
        return outputs
