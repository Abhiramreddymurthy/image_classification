# 🧠 Image Classification using Optimized AlexNet

This project implements an image classification system using an optimized version of the AlexNet architecture. It is trained on a dataset with 17 image classes and provides a Flask web interface to classify images uploaded by users — all running locally on your system.

---

## 📁 Project Structure

```
├── alexnet_model.py     # Defines the AlexNet model
├── train.py             # Training script for the model
├── inference.py         # Script for performing inference
├── app.py               # Flask web application
├── requirements.txt     # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Abhiramreddymurthy/image_classification
cd image_classification
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

If you don’t have `model.h5`, run the training script:

```bash
python train.py
```

This will generate and save `model.h5` (your trained model).

---

## 🚀 Running the Application Locally

Start the Flask web server:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

You can now upload an image and receive a predicted class from the model.

---

## 📊 Model Overview

- ✅ Architecture: Optimized AlexNet (from scratch)
- ✅ Classes: 17 image categories
- ✅ Libraries: TensorFlow, Keras, Flask, NumPy, Pillow

---

## 🧠 Applications

- Object detection and classification
- Educational deep learning demos
- Smart content filtering
- Image-based search and tagging

---

## 🔮 Future Enhancements

- Add transfer learning support (e.g., ResNet/EfficientNet)
- Create a desktop GUI version
- Add downloadable prediction results
- Improve UI with drag-and-drop support

---

## 👨‍💻 Author

**Abhiramreddymurthy**

---


