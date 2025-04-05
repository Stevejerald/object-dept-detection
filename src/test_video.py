import cv2
import yaml
from pathlib import Path
from video_handler import VideoHandler

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'app_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize video handler
    with VideoHandler(config) as video:
        # Create window for display
        cv2.namedWindow('Video Test', cv2.WINDOW_NORMAL)
        
        # Process frames
        for frame, frame_number in video.get_frames():
            # Display frame
            cv2.imshow('Video Test', frame)
            
            # Print frame information
            print(f"Processing frame {frame_number}, shape: {frame.shape}")
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 