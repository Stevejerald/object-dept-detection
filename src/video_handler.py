import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Generator

logger = logging.getLogger(__name__)

class VideoHandler:
    """Handles video input processing and frame management."""
    
    def __init__(self, config: dict):
        """
        Initialize the video handler.
        
        Args:
            config: Dictionary containing video configuration parameters
        """
        self.config = config['video']['input']
        self.processing_config = config['video']['processing']
        
        self.source = Path(self.config['source'])
        self.target_fps = self.config['fps']
        self.target_size = tuple(self.config['resize'])  # (width, height)
        
        self.cap = None
        self.frame_count = 0
        self.current_fps = 0
        
    def initialize(self) -> bool:
        """
        Initialize video capture.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.source.exists():
            logger.error(f"Video source not found: {self.source}")
            return False
        
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            logger.error("Failed to open video capture")
            return False
        
        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video loaded: {self.source}")
        logger.info(f"Original FPS: {self.original_fps}")
        logger.info(f"Frame count: {self.frame_count}")
        
        return True
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for depth estimation.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        if self.target_size != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, self.target_size)
        
        return frame
    
    def get_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Generator that yields preprocessed frames.
        
        Yields:
            Tuple containing:
                - Preprocessed frame
                - Frame number
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video capture not initialized")
            return
        
        frame_number = 0
        skip_frames = self.processing_config['skip_frames']
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.info("End of video reached")
                break
            
            # Skip frames if configured
            if skip_frames > 0 and frame_number % (skip_frames + 1) != 0:
                frame_number += 1
                continue
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            yield processed_frame, frame_number
            frame_number += 1
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
    
    def __enter__(self):
        """Context manager entry."""
        if self.initialize():
            return self
        raise RuntimeError("Failed to initialize video capture")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 