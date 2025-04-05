import cv2
import yaml
import numpy as np
from pathlib import Path
from video_handler import VideoHandler
from model_handler import DepthEstimator
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configurations
        config_dir = Path(__file__).parent.parent / 'config'
        logger.info("Loading configuration files...")
        
        with open(config_dir / 'app_config.yaml', 'r') as f:
            app_config = yaml.safe_load(f)
        
        with open(config_dir / 'model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        
        logger.info("Initializing depth estimator...")
        # Initialize depth estimator
        with DepthEstimator(model_config) as depth_estimator:
            logger.info("Initializing video handler...")
            # Initialize video handler
            with VideoHandler(app_config) as video:
                # Create windows for display
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
                
                logger.info("Starting frame processing...")
                # Process frames
                for frame, frame_number in video.get_frames():
                    # Estimate depth
                    depth_map = depth_estimator.estimate_depth(frame)
                    
                    if depth_map is not None:
                        # Normalize depth map for visualization
                        depth_colored = plt.cm.plasma(depth_map)[:, :, :3]  # Remove alpha channel
                        depth_colored = (depth_colored * 255).astype(np.uint8)
                        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                        
                        # Display frames
                        cv2.imshow('Original', frame)
                        cv2.imshow('Depth', depth_colored)
                        
                        logger.info(f"Processing frame {frame_number}, depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")
                        
                        # Break if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        logger.error(f"Failed to process frame {frame_number}")
        
        # Cleanup
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 