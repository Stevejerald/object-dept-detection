# Real-Time Monocular Depth Estimation System

A Python-based system for **real-time monocular depth estimation** using the MiDaS model with support for GPU acceleration, bounding box detection of nearby objects, and high-resolution video display.

## 🚀 Overview

This project implements a real-time monocular depth estimation system that can process video input and generate depth maps.  
It uses the **MiDaS model (Intel)** for depth estimation and provides an extensible interface for video processing.

Recent updates include:
- GPU/CPU auto-detection for improved performance
- Frame skipping strategy for near real-time results
- Visualization of depth maps in **Plasma colormap**
- **Bounding boxes** around nearby objects based on depth thresholds
- Configurable display width for large monitors

## ✨ Features

- Real-time video processing with depth estimation
- Depth maps using MiDaS
- Bounding box detection for nearby objects
- Configurable:
  - Video input parameters
  - Model device (CPU/GPU)
  - Display width
- Depth visualization in color
- Modular project structure for easy extension

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Stevejerald/object-dept-detection.git
   cd object-dept-detection
````

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate # On Linux/Mac
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Run the Main Application

To start the depth estimation pipeline with real-time bounding boxes:

```bash
python src/main.py
```

### Test Video Processing

```bash
python src/test_video.py
```

### Test Depth Estimation

```bash
python src/test_depth.py
```

---

## ⚙️ Configuration

The system uses two YAML configuration files:

* `config/app_config.yaml` → Application settings (video source, resolution, etc.)
* `config/model_config.yaml` → Model settings (weights, backend, device)

---

## 📂 Project Structure

```
.
├── config/               # Configuration files
│   ├── app_config.yaml
│   └── model_config.yaml
├── data/                 # Data directory
│   ├── videos/           # Video input files
│   └── sample_outputs/   # Sample output files
├── models/               # Model directory
│   └── weights/          # Model weights
├── src/                  # Source code
│   ├── main.py           # Main entry point (depth + bounding boxes)
│   ├── video_handler.py  # Video processing
│   ├── model_handler.py  # Model handling
│   ├── test_video.py     # Video test script
│   └── test_depth.py     # Depth test script
├── tests/                # Unit/integration tests
├── utils/                # Utility functions
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── DEVELOPMENT_PLAN.md   # Future development roadmap
```

---

## 📌 Development Roadmap

See [DEVELOPMENT_PLAN.md] (DEVELOPMENT_PLAN.md) for detailed future improvements and planned features.

---

## Acknowledgments

* [Intel MiDaS model](https://github.com/isl-org/MiDaS)
* [OpenCV](https://opencv.org/) for video processing
* [PyTorch](https://pytorch.org/) for deep learning

```