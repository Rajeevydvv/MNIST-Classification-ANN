# ✍️ MNIST Handwritten Digit Classification using ANN

A Deep Learning project that classifies handwritten digits (0–9) from the MNIST dataset using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The project also includes a Streamlit web application where users can draw digits on a canvas and receive real-time predictions.

---

## 🚀 Project Overview

Handwritten digit recognition is one of the most fundamental problems in Computer Vision and Deep Learning. In this project, an Artificial Neural Network is trained on the MNIST dataset to recognize handwritten digits with high accuracy.

The trained model is deployed using Streamlit, allowing users to interact with the model through a browser-based drawing canvas.

---

## 📂 Project Structure

```text
MNIST-Classification-ANN/
│
├── app.py                  # Streamlit Application
├── requirements.txt        # Project Dependencies
├── mnist_ann.keras         # Trained ANN Model
├── experiments.ipynb       # Model Training Notebook
├── README.md
```

---

## 🧠 Model Architecture

The ANN consists of:

```text
Input Image (28x28)
        ↓
Flatten Layer
        ↓
Dense (128 neurons, ReLU)
        ↓
Dense (10 neurons, Softmax)
```

### Architecture Summary

* Input Shape: 28 × 28
* Flatten Layer
* Hidden Layer: 128 Neurons (ReLU)
* Output Layer: 10 Neurons (Softmax)
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Evaluation Metric: Accuracy

---

## 📊 Dataset

### MNIST Dataset

The MNIST dataset contains handwritten digits from 0 to 9.

Dataset Statistics:

| Split    | Samples |
| -------- | ------: |
| Training |  60,000 |
| Testing  |  10,000 |

Image Characteristics:

* Grayscale Images
* Resolution: 28 × 28
* 10 Classes (Digits 0–9)

MNIST is one of the most widely used benchmark datasets in Deep Learning and Computer Vision.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

1. Normalize pixel values from 0–255 to 0–1
2. Reshape images into the required format
3. Feed images into the ANN model

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

---

## 🏋️ Model Training

Model was trained using:

```python
optimizer='adam'
loss='sparse_categorical_crossentropy'
metrics=['accuracy']
```

Training Process:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2
)
```

---

## 📈 Results

The model successfully learns handwritten digit patterns and achieves strong classification performance on the MNIST test dataset.

Metrics Evaluated:

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Loss Curves

---

## 🎨 Streamlit Application

The project includes a Streamlit web application where users can:

* Draw handwritten digits
* Get real-time predictions
* View prediction confidence
* Interact with the trained ANN model

### Features

✅ Interactive Drawing Canvas

✅ Real-time Prediction

✅ Confidence Score

---

## 🖥️ Run Locally

### Clone Repository

```bash
git clone https://github.com/Rajeevydvv/MNIST-Classification-ANN.git

cd MNIST-Classification-ANN
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit
* Pillow

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

* Artificial Neural Networks (ANN)
* Forward Propagation
* Backpropagation
* Activation Functions
* Multi-Class Classification
* TensorFlow/Keras Workflow
* Model Deployment with Streamlit
* Deep Learning Project Structure

---

## 🔮 Future Improvements

* Convolutional Neural Network (CNN) implementation
* Improved canvas preprocessing
* OpenCV digit centering and normalization
* Streamlit Cloud deployment
* Model comparison (ANN vs CNN)

---

## 👨‍💻 Author

**Rajeev Yadav**

GitHub: https://github.com/Rajeevydvv

---

### ⭐ If you found this project useful, consider giving the repository a star.
