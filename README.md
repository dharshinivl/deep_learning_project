# 🔍 Deepfake Detection System

A comprehensive AI-powered deepfake detection system with modern web interface, designed for journalists, cybersecurity professionals, and the general public.

## 🎯 Features

- **High Accuracy**: 95.8% accuracy on Celeb-DF dataset
- **Real-time Analysis**: Fast inference with GPU acceleration
- **Visual Explainability**: Grad-CAM heatmaps showing suspicious regions
- **Multi-format Support**: Images (PNG, JPG) and Videos (MP4, AVI, MKV)
- **Modern UI**: Responsive, drag-and-drop interface
- **API-first**: RESTful API for integration

## 📊 Model Performance

- **Training Dataset**: Celeb-DF (3,115 balanced videos)
- **Architecture**: EfficientNetV2-S with custom binary head
- **Test Results**:
  - Accuracy: 95.82%
  - Precision: 94.85%
  - Recall: 99.55%
  - F1-Score: 97.14%

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd DL-PROJECT
```

2. **Install backend dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

3. **Start the API server**:
```bash
python app.py
```

4. **Open the frontend**:
   - Open `frontend/index.html` in your browser
   - Or serve with a local server: `python -m http.server 3000` in the frontend directory

### Usage

1. **Web Interface**:
   - Open the frontend in your browser
   - Select Image or Video mode
   - Upload media files via drag-and-drop or file picker
   - Optionally enable heatmap visualization
   - View results with confidence scores

2. **API Endpoints**:
   - `POST /api/predict-image`: Analyze images
   - `POST /api/predict-video`: Analyze videos  
   - `GET /api/health`: Check system status

## 🏗️ Project Structure

```
DL-PROJECT/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── inference.py        # Model inference logic
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   └── index.html          # Modern web interface
├── outputs/
│   ├── best_model.pth      # Trained model weights
│   └── predictions_log.csv # Prediction logs
├── data/
│   └── DFDC/              # Dataset (Celeb-DF)
└── train.py               # Training script
```

## 🔧 Configuration

### Environment Variables

- `WEIGHTS_PATH`: Path to model weights (default: `../outputs/best_model.pth`)
- `THRESHOLD`: Decision threshold (default: `0.5`)
- `MAX_UPLOAD_MB`: Max file size (default: `150`)
- `FRAME_SAMPLES`: Video frames to analyze (default: `8`)
- `RATE_LIMIT_PER_MIN`: API rate limit (default: `60`)

### API Configuration

```bash
# Example configuration
export THRESHOLD=0.6
export MAX_UPLOAD_MB=300
export FRAME_SAMPLES=16
python app.py
```

## 📈 API Examples

### Image Analysis
```bash
curl -X POST "http://127.0.0.1:8000/api/predict-image" \
  -F "file=@test_image.jpg" \
  -F "heatmap=true"
```

### Video Analysis
```bash
curl -X POST "http://127.0.0.1:8000/api/predict-video" \
  -F "file=@test_video.mp4"
```

### Response Format
```json
{
  "prob": 0.915,
  "label": 1,
  "threshold": 0.5,
  "confidence": "high",
  "frames": 8,
  "heatmap_b64_jpg": "base64_encoded_image"
}
```

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop and mobile
- **Drag & Drop**: Easy file uploads
- **Real-time Feedback**: Loading states and progress indicators
- **Visual Results**: Confidence bars and probability displays
- **Heatmap Visualization**: Grad-CAM explanations
- **Media Type Switching**: Toggle between image and video modes

## 🔬 Technical Details

### Model Architecture
- **Backbone**: EfficientNetV2-S (pretrained on ImageNet)
- **Head**: Dropout (0.3) + Linear layer for binary classification
- **Input**: 256×256 face crops
- **Preprocessing**: Face detection, normalization, augmentation

### Face Detection
- **Primary**: OpenCV Haar cascades
- **Fallback**: Center-crop when detection fails
- **Parameters**: scaleFactor=1.05, minNeighbors=3

### Explainability
- **Grad-CAM**: Visual attention maps
- **Target Layer**: EfficientNetV2-S features[-1]
- **Visualization**: Jet colormap overlay

## 🚀 Deployment

### Local Development
```bash
# Backend
cd backend && python app.py

# Frontend (optional server)
cd frontend && python -m http.server 3000
```

### Production Considerations
- Use Gunicorn/Uvicorn with multiple workers
- Implement proper CORS policies
- Add authentication and rate limiting
- Use a reverse proxy (nginx)
- Set up monitoring and logging

## 📊 Monitoring

- **Prediction Logs**: CSV format in `outputs/predictions_log.csv`
- **Health Check**: `/api/health` endpoint
- **Rate Limiting**: Per-IP request limits
- **Error Handling**: Comprehensive error responses

## 🔮 Future Improvements

- [ ] Multi-modal analysis (audio + video)
- [ ] Advanced face detection (MediaPipe, RetinaFace)
- [ ] Temporal consistency analysis
- [ ] Batch processing capabilities
- [ ] User authentication and history
- [ ] Mobile app development
- [ ] Real-time streaming analysis

## 📝 License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when using deepfake detection technology.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Built with ❤️ using PyTorch, FastAPI, and modern web technologies**



