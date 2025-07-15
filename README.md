# 🐾 Animal Image Classifier 🧠

A deep learning-based image classification project that detects animal classes (e.g., cats, dogs, snakes) using a CNN model trained in TensorFlow and served using FastAPI.

---

## 📁 Project Structure
```bash
  image_classifier/
├── app/
│ └── model/
│ ├── cnn_animal_model.h5 # Trained CNN model
│ └── label_encoder.pkl # Label encoder for class names
│
├── dataset/
│ └── Animals/
│ ├── cats/
│ ├── dogs/
│ └── snakes/
│
├── static/
│ └── upload.html # Frontend HTML form for uploading images
│
├── train_cnn_model.py # Model training script (modular)
├── main.py # FastAPI server for prediction and training
├── requirements.txt # Python dependencies
└── README.md # Documentation
```
---
## 🚀 Features

- 🧠 Trains a CNN using TensorFlow/Keras
- 🔁 Automatically retrains when new image is uploaded
- 🖼️ Classifies animal images using a trained model
- 🌐 FastAPI backend with image upload and prediction
- 🧪 Label encoding for categorical class support
- 📊 Uses `ImageDataGenerator` for data augmentation

---

## 🛠️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/saugatpoudel100/image_classifier.git
cd image_classifier
```

Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
```

Install dependencies

```bash
pip install -r requirements.txt
```
Prepare the dataset

Organize your dataset as:
```bash

dataset/Animals/
├── cats/
├── dogs/
└── snakes/
```
Add JPG/PNG images in respective folders.

## Train the model
```bash
python train_cnn_model.py
```
Run the server
```bash
uvicorn main:app --reload
```
Access the Web Interface

Open browser:
http://127.0.0.1:8000

## 🧪 API Endpoints
GET /
→ HTML form for image upload

POST /predict
→ Upload an image and get prediction result

POST /upload_and_train
→ Upload new image to a class and retrain the model
Form Fields:

file: image file

label: category (must match existing or create new folder)

---

## 📈 Model Details
CNN Layers:

Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout → Output

Activation: ReLU (hidden), Softmax (output)

Optimizer: Adam

Loss: Categorical Crossentropy

---

## 🧾 Requirements
See requirements.txt.
Main libraries:

tensorflow

scikit-learn

scikit-image

fastapi

uvicorn

joblib

---
## 📷 Sample Result
json:
```bash
POST /predict
{
  "prediction": "cats"
}
```
---
## 🤝 Contributing
Feel free to fork and submit PRs for improvements, bug fixes, or new features!
 ---
## 📄 License
MIT License. See LICENSE for details.









