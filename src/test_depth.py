import cv2
import yaml
import numpy as np
from pathlib import Path
from video_handler import VideoHandler
from model_handler import DepthEstimator
import logging
import torch  # <-- for GPU check

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many frames to skip between depth inferences
SKIP_EVERY = 8

# Desired output width for display
DISPLAY_WIDTH = 2500

# Threshold (0-255) on normalized depth to consider "nearby".
# Lower threshold -> only very close things, higher -> more objects considered nearby.
CLOSE_THRESH = 60

# Minimum area for a contour to be considered (filter noise)
MIN_BOX_AREA = 1500  # adjust depending on resolution and object size

# Morphology kernel sizes for cleaning the mask
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def resize_with_aspect(frame, width):
    """Resize frame keeping aspect ratio given target width."""
    h, w = frame.shape[:2]
    new_height = int(h * (width / w))
    return cv2.resize(frame, (width, new_height))


def extract_nearby_bboxes_from_depth(depth_map):
    """
    Given a 2D depth_map (float), return bounding boxes for nearby objects.
    Returns list of (x, y, w, h) in coordinates matching depth_map shape.
    """
    # Normalize depth to 0-255 uint8 (min -> 0, max -> 255)
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # Invert-threshold so that small depth values (near objects) become white
    _, mask = cv2.threshold(depth_norm, CLOSE_THRESH, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= MIN_BOX_AREA:
            boxes.append((x, y, w, h))

    return boxes, depth_norm  # return normalized depth for visualization if needed


def main():
    try:
        # Load configurations
        config_dir = Path(__file__).parent.parent / 'config'
        logger.info("Loading configuration files...")

        with open(config_dir / 'app_config.yaml', 'r') as f:
            app_config = yaml.safe_load(f)

        with open(config_dir / 'model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)

        # ---- (4) Use GPU if available ----
        use_cuda = torch.cuda.is_available()
        chosen_device = "cuda" if use_cuda else "cpu"
        model_config["device"] = chosen_device
        logger.info(f"Selecting device: {chosen_device}")
        if use_cuda:
            torch.backends.cudnn.benchmark = True

        logger.info("Initializing depth estimator...")
        with DepthEstimator(model_config) as depth_estimator:
            logger.info("Initializing video handler...")
            with VideoHandler(app_config) as video:
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

                logger.info("Starting frame processing...")
                last_depth_colored = None
                last_boxes = []

                for i, (frame, frame_number) in enumerate(video.get_frames()):
                    # Show original resized frame every loop for smooth playback
                    orig_h, orig_w = frame.shape[:2]
                    scale = DISPLAY_WIDTH / orig_w  # uniform scale factor for x and y
                    frame_resized = resize_with_aspect(frame, DISPLAY_WIDTH)

                    # Draw any last detected boxes on the resized original for continuous feedback
                    display_frame = frame_resized.copy()
                    if last_boxes:
                        for (bx, by, bw, bh) in last_boxes:
                            # scale coordinates to resized display
                            sx = int(bx * scale)
                            sy = int(by * scale)
                            sw = int(bw * scale)
                            sh = int(bh * scale)
                            cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 3)
                            # Optionally show label and (approx) distance indicator
                            cv2.putText(display_frame, "Nearby", (sx, sy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    cv2.imshow('Original', display_frame)

                    # Only compute depth and boxes on selected frames (skip for speed)
                    if i % (SKIP_EVERY + 1) == 0:
                        depth_map = depth_estimator.estimate_depth(frame)
                        if depth_map is not None:
                            # Get boxes in original frame coords and normalized depth for coloring
                            boxes, depth_norm = extract_nearby_bboxes_from_depth(depth_map)

                            # Create fast color map for depth visualization (uint8)
                            depth_vis = depth_norm.copy()
                            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
                            depth_vis = resize_with_aspect(depth_vis, DISPLAY_WIDTH)

                            last_depth_colored = depth_vis
                            last_boxes = boxes  # update boxes (in original coords)
                            logger.info(
                                f"Processed frame {frame_number} (i={i}), "
                                f"found {len(boxes)} nearby boxes, depth range: {depth_map.min():.2f} - {depth_map.max():.2f}"
                            )
                        else:
                            logger.error(f"Failed to process frame {frame_number}")
                    # show the most recent depth visualization
                    if last_depth_colored is not None:
                        cv2.imshow('Depth', last_depth_colored)

                    # Exit on 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
