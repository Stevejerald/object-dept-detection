# Real-Time Monocular Depth Estimation System

A Python-based system for real-time monocular depth estimation using the MiDaS model.

## Overview

This project implements a real-time monocular depth estimation system that can process video input and generate depth maps. It uses the MiDaS model from Intel for depth estimation and provides a simple interface for video processing.

## Features

- Real-time video processing
- Depth estimation using MiDaS model
- Configurable video input and processing parameters
- Visualization of depth maps

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/real-time-depth-estimation.git
cd real-time-depth-estimation
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Testing Video Processing

To test the video processing functionality:
```
python src/test_video.py
```

### Testing Depth Estimation

To test the depth estimation functionality:
```
python src/test_depth.py
```

## Configuration

The system uses two configuration files:
- `config/app_config.yaml`: Application configuration
- `config/model_config.yaml`: Model configuration

## Project Structure

```
.
├── config/             # Configuration files
├── data/               # Data directory
│   ├── videos/         # Video input files
│   └── sample_outputs/ # Sample output files
├── models/             # Model directory
│   └── weights/        # Model weights
├── src/                # Source code
│   ├── main.py         # Main entry point
│   ├── video_handler.py # Video processing
│   ├── model_handler.py # Model handling
│   ├── test_video.py   # Video test script
│   └── test_depth.py   # Depth test script
├── tests/              # Test files
├── utils/              # Utility functions
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## License

[Your chosen license]

## Acknowledgments

- MiDaS model from Intel
- OpenCV for video processing
- PyTorch for deep learning 