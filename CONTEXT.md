# Real-Time Monocular Depth Estimation System

## Project Overview
This project implements a real-time monocular depth estimation system using deep learning. While originally designed for deployment on NVIDIA Jetson Nano with a live camera feed, this development version runs on a standard laptop using pre-recorded traffic videos for testing and development purposes.

## Objectives
- Implement real-time depth estimation using a single RGB video input
- Generate accurate depth maps from 2D video frames
- Provide distance measurements and proximity alerts
- Achieve optimal performance for real-time processing

## Technical Architecture

### Hardware Setup (Development Environment)
- Standard laptop/desktop computer
- GPU (if available) for faster processing
- Display for visualization

### Software Components
1. **Core Dependencies**
   - Python 3.8+
   - PyTorch and TorchVision
   - OpenCV for video processing
   - NumPy for numerical computations

2. **Deep Learning Models**
   - MiDaS (Towards Robust Monocular Depth Estimation)
   - FastDepth (optimized for real-time processing)
   - MonoDepth2 (specialized for driving scenarios)

3. **Data Pipeline**
   - Video input processing
   - Frame preprocessing
   - Depth map generation
   - Distance calculation
   - Alert system simulation

### Datasets
- KITTI Dataset (driving scenes)
- NYUv2 Dataset (indoor scenes)
- Custom traffic videos for testing

## System Workflow
1. **Video Input**
   - Load and process video frames
   - Implement frame rate control
   - Handle video stream management

2. **Depth Estimation**
   - Pre-process input frames
   - Run inference using selected model
   - Generate depth maps
   - Post-process results

3. **Distance Analysis**
   - Convert depth maps to distance measurements
   - Implement proximity detection
   - Generate safety alerts based on thresholds:
     - Safe zone
     - Caution zone
     - Danger zone

4. **Visualization**
   - Display original video feed
   - Show depth map visualization
   - Render distance measurements
   - Display alert status

## Performance Metrics
- Inference time per frame
- Depth estimation accuracy
- Frame processing rate
- Memory usage

## Future Enhancements
1. **Model Optimization**
   - Model quantization
   - TensorRT integration
   - Performance profiling and optimization

2. **Feature Extensions**
   - Multi-object tracking
   - Distance estimation for specific objects
   - Integration with other sensor data

3. **Deployment**
   - Migration to Jetson Nano
   - Live camera integration
   - Hardware alert system implementation

## Project Structure
```
project/
├── src/              # Source code
├── data/             # Dataset and video files
├── models/           # Pre-trained models
├── utils/            # Utility functions
├── tests/            # Unit tests
└── config/           # Configuration files
```

## Getting Started
Detailed setup and running instructions will be provided in the README.md file.

## References
1. MiDaS: https://github.com/intel-isl/MiDaS
2. MonoDepth2: https://github.com/nianticlabs/monodepth2
3. KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
4. NYU Depth V2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html 