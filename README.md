# vehicledamagedetection
AI-Based Vehicle Damage Detection and Classification using Deep Learning
# 🚗 Vehicle Damage Detection

AI-Based Vehicle Damage Detection and Classification using Deep Learning

## 📊 Project Overview

An automated system that detects and classifies vehicle damage from images using transfer learning with EfficientNetB0. Achieves **89%+ accuracy** in identifying damage types including dents, scratches, cracks, broken glass, and missing parts.

## ✨ Key Features

- 🎯 **High Accuracy:** 89.3% validation accuracy
- ⚡ **Fast Inference:** <0.5 seconds per image
- 🔧 **Multiple Damage Types:** Detects 6-9 different damage categories
- 📊 **Transfer Learning:** Uses pre-trained EfficientNetB0
- 💾 **Small Model Size:** Only 21MB

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Google Colab (recommended for GPU)

### Installation
```bash
pip install tensorflow pillow numpy pandas matplotlib seaborn roboflow
```

### Download Dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="rf_8bENBieYT9MRd9VfkGjbp7sDtqU2")
project = rf.workspace("car-damage-kadad").project("car-damage-images")
dataset = project.version(1).download("folder")
```

### Train Model

1. Open `Vehicle_Damage_Detection_Complete.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Wait ~60 minutes for training to complete

### Make Predictions
```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('best_model.h5')

# Load image
img = Image.open('car.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_names = ['dent', 'scratch', 'crack', 'broken_glass', 'missing_part']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")
```

## 📈 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | 89.3% |
| Precision | 89.1% |
| Recall | 88.6% |
| F1-Score | 88.8% |
| Training Time | ~45 min (GPU) |

## 🏗️ Model Architecture

- **Base Model:** EfficientNetB0 (pre-trained on ImageNet)
- **Custom Layers:** Dense(512) → Dense(256) → Output
- **Total Parameters:** 5.3M
- **Trainable Parameters:** 1.3M

## 📊 Dataset

- **Source:** Roboflow Universe - Car Damage Images
- **Total Images:** ~3,000 labeled images
- **Classes:** 6-9 damage types
- **Split:** 70% Train / 20% Val / 10% Test

**Damage Types:**
- Dent
- Scratch
- Crack
- Broken Glass
- Missing Part
- Tire Flat
- Paint Damage

## 💡 Real-World Applications

- 🏢 **Insurance:** Automated claim processing
- 🚗 **Car Rentals:** Pre/post-rental inspection
- 🔧 **Repair Shops:** Damage triage and cost estimation
- 🏭 **Manufacturing:** Quality control and PDI

## 📁 Project Structure
```
├── notebooks/
│   └── Vehicle_Damage_Detection_Complete.ipynb
├── models/
│   └── best_model.h5
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── classification_report.txt
└── README.md
```

## 🛠️ Tech Stack

- **Python** 3.8+
- **TensorFlow** 2.x
- **Keras** - Deep learning API
- **EfficientNetB0** - Base model
- **NumPy, Pandas** - Data processing
- **Matplotlib, Seaborn** - Visualization
- **Roboflow** - Dataset management

## 👥 Team

**Course:** AAI-521 Computer Vision  
**Institution:** University of San Diego  
**Semester:** Fall 2024

| Member | Role |
|--------|------|
| Pavan Kumar Kallakuri | Data Collection & EDA |
| Sajesh Kariadan | Model Development |
| Nishchal P | Evaluation & Testing |

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- University of San Diego - AAI Program
- Roboflow - Dataset hosting
- TensorFlow Team - Framework
- Google Colab - Free GPU access

## 📧 Contact

**GitHub:** [pavankallakuri9](https://github.com/pavankallakuri9)  
**Project Link:** [Vehicle Damage Detection](https://github.com/pavankallakuri9/vehicledamagedetection)

---

⭐ **Star this repo if you found it helpful!**
