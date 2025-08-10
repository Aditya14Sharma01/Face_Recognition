# Human Face Recognition & Analysis System

A comprehensive Python-based computer vision system for real-time human face recognition, age estimation, gender detection, and emotion analysis. This project provides both CPU and GPU-optimized versions for different hardware configurations.

## 🚀 Features

- **Real-time Face Detection**: Advanced face detection using RetinaFace backend
- **Multi-attribute Analysis**: Age, gender, and emotion recognition
- **Dual Performance Modes**: CPU-optimized and GPU-accelerated versions
- **High-Resolution Support**: 1280x720 resolution for enhanced accuracy
- **Live Webcam Integration**: Real-time analysis with webcam feed
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📁 Repository Components

### Core Scripts
- **`gpu-Recog.py`** - GPU-accelerated version using PyTorch CUDA support
- **`cpu-Recog.py`** - CPU-optimized version with threading for better performance
- **`human recognition.py`** - Main recognition module (placeholder for future enhancements)
- **`RunTest.py`** - System testing and validation script

### Model Files
- **`models/age_net.caffemodel`** - Pre-trained age estimation model
- **`models/gender_net.caffemodel`** - Pre-trained gender classification model

## 📸 Photos

Sample photos for testing the face recognition system:
![Photos/Photo (1).png](Photos/Photo (1).png)
- **Photo (1).png** - Sample image for testing face detection and analysis
![Photos/Photo (2).png](Photos/Photo (2).png)
- **Photo (2).png** - Additional sample image for validation

## 🛠️ Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- CUDA-compatible GPU (for GPU version)
- At least 4GB RAM

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Hard.git
cd Hard
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install opencv-python
pip install deepface
pip install torch torchvision torchaudio
pip install numpy
```

### Step 4: Download Pre-trained Models
The Caffe models should already be included in the `models/` folder. If not, download them:
```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download models (you'll need to provide the actual download links)
# age_net.caffemodel
# gender_net.caffemodel
```

## 🚀 Usage

### Testing System Components
First, test if your system is properly configured:
```bash
python RunTest.py
```

This will verify:
- OpenCV installation
- Cascade classifiers availability
- DNN module functionality
- Webcam accessibility

### Running CPU Version
```bash
python cpu-Recog.py
```

**Features:**
- Threaded analysis for better performance
- Configurable refresh intervals
- Resizable window support
- Press 'q' to quit

### Running GPU Version
```bash
python gpu-Recog.py
```

**Features:**
- CUDA acceleration (if available)
- Real-time analysis every 10 frames
- Enhanced brightness/contrast
- Press ESC or close window to quit

## ⚙️ Configuration

### Performance Tuning
- **CPU Version**: Adjust `REFRESH_INTERVAL` in `cpu-Recog.py` for analysis frequency
- **GPU Version**: Modify frame analysis frequency (currently every 10 frames)
- **Resolution**: Change `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` for different resolutions

### Backend Selection
Both versions use RetinaFace as the detector backend for optimal accuracy. You can change this in the `DETECTOR_BACKEND` variable.

## 🔧 Troubleshooting

### Common Issues

1. **Webcam Not Working**
   - Check if webcam is accessible by other applications
   - Try changing camera index (0, 1, 2) in `cv2.VideoCapture()`

2. **CUDA Not Available (GPU Version)**
   - Install CUDA toolkit
   - Install PyTorch with CUDA support
   - Check GPU compatibility

3. **Model Loading Errors**
   - Ensure models are in the `models/` folder
   - Check file permissions
   - Verify model file integrity

4. **Performance Issues**
   - Use CPU version for older hardware
   - Reduce resolution for better performance
   - Increase refresh intervals

### Performance Optimization
- **CPU**: Use threading and optimize refresh intervals
- **GPU**: Ensure CUDA is properly configured
- **Memory**: Close other applications to free up RAM

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.7     | 3.9+       |
| RAM       | 4GB     | 8GB+       |
| Storage   | 2GB     | 5GB+       |
| GPU       | None    | CUDA 11.0+ |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- DeepFace library for facial analysis capabilities
- PyTorch team for GPU acceleration support
- RetinaFace for accurate face detection

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Ensure all dependencies are properly installed

## 🔄 Updates

Stay updated with the latest features and improvements by:
- Starring this repository
- Watching for updates
- Checking the releases page

---

**Note**: This system is designed for educational and research purposes. Always respect privacy and obtain consent when analyzing faces in real-world applications.
