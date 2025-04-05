import torch
import cv2
import numpy as np
from pathlib import Path
import logging
import urllib.request
import os

logger = logging.getLogger(__name__)

class DepthEstimator:
    """Handles depth estimation using MiDaS model."""
    
    def __init__(self, config: dict):
        """
        Initialize the depth estimator.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config['model']
        self.preprocessing = config['preprocessing']
        self.postprocessing = config['postprocessing']
        
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        logger.info(f"Using device: {self.device}")
    
    def initialize(self) -> bool:
        """
        Initialize the model and transforms.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize model using torch hub
            if self.config['name'] == "MiDaS":
                model_type = "DPT_Large" if self.config['version'] == "DPT_Large" else "MiDaS_small"
                
                # Load model and transforms
                self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                self.model.to(self.device)
                self.model.eval()
                
                # Load transforms
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform
                
                logger.info(f"Model {model_type} initialized successfully")
                return True
            else:
                logger.error(f"Unsupported model: {self.config['name']}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Apply MiDaS transform
        tensor = self.transform(frame)
        
        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from input frame.
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            
        Returns:
            Depth map as numpy array
        """
        if self.model is None:
            logger.error("Model not initialized")
            return None
        
        try:
            # Preprocess frame
            img = self.preprocess(frame)
            img = img.to(self.device)
            
            with torch.no_grad():
                # Forward pass
                prediction = self.model(img)
                
                # Ensure prediction has the right shape
                if len(prediction.shape) == 3:
                    prediction = prediction.unsqueeze(1)
                
                # Normalize prediction
                prediction = torch.nn.functional.interpolate(
                    prediction,
                    size=frame.shape[:2],
                    mode=self.preprocessing['resize_mode'],
                    align_corners=False,
                ).squeeze()
                
                depth = prediction.cpu().numpy()
                
                # Scale depth values
                depth = self.postprocessing['depth_scale'] * (depth - depth.min()) / (depth.max() - depth.min())
                
                return depth
                
        except Exception as e:
            logger.error(f"Error in depth estimation: {str(e)}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize model")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up resources
        if self.model is None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources released") 