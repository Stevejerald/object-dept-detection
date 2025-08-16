import cv2
import yaml
import numpy as np
from pathlib import Path
from video_handler import VideoHandler
from model_handler import DepthEstimator
import logging
import torch  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SKIP_EVERY = 8

DISPLAY_WIDTH = 2500

CLOSE_THRESH = 60

MIN_BOX_AREA = 1500 

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
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    
    _, mask = cv2.threshold(depth_norm, CLOSE_THRESH, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= MIN_BOX_AREA:
            boxes.append((x, y, w, h))

    return boxes, depth_norm


def main():
    try:
        config_dir = Path(__file__).parent.parent / 'config'
        logger.info("Loading configuration files...")

        with open(config_dir / 'app_config.yaml', 'r') as f:
            app_config = yaml.safe_load(f)

        with open(config_dir / 'model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)

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
                    orig_h, orig_w = frame.shape[:2]
                    scale = DISPLAY_WIDTH / orig_w 
                    frame_resized = resize_with_aspect(frame, DISPLAY_WIDTH)

                    display_frame = frame_resized.copy()
                    if last_boxes:
                        for (bx, by, bw, bh) in last_boxes:
                            sx = int(bx * scale)
                            sy = int(by * scale)
                            sw = int(bw * scale)
                            sh = int(bh * scale)
                            cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 3)
                            cv2.putText(display_frame, "Nearby", (sx, sy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    cv2.imshow('Original', display_frame)

                    if i % (SKIP_EVERY + 1) == 0:
                        depth_map = depth_estimator.estimate_depth(frame)
                        if depth_map is not None:
                            boxes, depth_norm = extract_nearby_bboxes_from_depth(depth_map)

                            depth_vis = depth_norm.copy()
                            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
                            depth_vis = resize_with_aspect(depth_vis, DISPLAY_WIDTH)

                            last_depth_colored = depth_vis
                            last_boxes = boxes
                            logger.info(
                                f"Processed frame {frame_number} (i={i}), "
                                f"found {len(boxes)} nearby boxes, depth range: {depth_map.min():.2f} - {depth_map.max():.2f}"
                            )
                        else:
                            logger.error(f"Failed to process frame {frame_number}")
                    if last_depth_colored is not None:
                        cv2.imshow('Depth', last_depth_colored)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
