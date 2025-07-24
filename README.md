
# 🧠 Brain Tumor MRI Image Classification

## 📌 Problem Statement

This project aims to develop a **deep learning-based solution** for classifying brain MRI images into multiple categories according to tumor type. It involves:

- Building a **custom CNN model** from scratch
- Enhancing performance through **transfer learning** (MobileNetV2)
- Deploying a **user-friendly Streamlit app** for real-time predictions

---

## 🗂️ Project Overview

This repository contains code and resources to train a deep learning model to classify MRI scans into:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

### 🔍 Key Features

- 🧹 **Data Preprocessing & Augmentation**
- 🧠 **Custom CNN Architecture**
- 🤖 **Transfer Learning with MobileNetV2**
- 📊 **Model Evaluation** using accuracy, precision, recall, F1-score, and confusion matrix
- 🌐 **Interactive Streamlit Web App** for predictions

---

## 📁 Dataset

The dataset consists of brain MRI images categorized into:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

### 📊 Dataset Split & Distribution

| Class         | Training Images | Validation Images | Test Images |
|---------------|------------------|--------------------|-------------|
| Glioma        | 564              | 161                | 80          |
| Meningioma    | 358              | 125                | 63          |
| No Tumor      | 336              | 99                 | 49          |
| Pituitary     | 438              | 118                | 54          |
| **Total**     | **1696**         | **503**            | **246**     |

> ⚖️ *Note: A moderate class imbalance exists, handled via class weighting during training.*

---

## 🧠 Model Architectures

### 🔨 1. Custom CNN (From Scratch)

```python
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

---

### 📦 2. Transfer Learning (MobileNetV2)

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

tl_model = Model(inputs=base_model.input, outputs=predictions)
```

> 🔍 *EfficientNetB0 was also explored but not fully implemented.*

---

## 📈 Evaluation & Results

### 🧪 Custom CNN - Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.71      | 0.91   | 0.80     | 80      |
| Meningioma   | 0.00      | 0.00   | 0.00     | 63      |
| No Tumor     | 0.77      | 0.73   | 0.75     | 49      |
| Pituitary    | 0.54      | 0.96   | 0.69     | 54      |

- **Accuracy:** 0.65  
- **Macro Avg:** Precision 0.50, Recall 0.65, F1-score 0.56  
- **Weighted Avg:** Precision 0.50, Recall 0.65, F1-score 0.56  

---

### 🧪 MobileNetV2 - Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.88      | 0.81   | 0.84     | 80      |
| Meningioma   | 0.76      | 0.73   | 0.75     | 63      |
| No Tumor     | 0.92      | 0.90   | 0.91     | 49      |
| Pituitary    | 0.86      | 0.94   | 0.90     | 54      |

- **Accuracy:** 0.85  
- **Macro Avg:** Precision 0.85, Recall 0.85, F1-score 0.85  
- **Weighted Avg:** Precision 0.85, Recall 0.85, F1-score 0.85  

---

## 🧪 Confusion Matrix (MobileNetV2)

![Confusion Matrix](assets/confusion_matrix.png)

---

## 📊 Training Curves



### MobileNetV2 Loss & Accuracy

| ![Train vs Val Loss](assets/train_val_loss.png) | ![Train vs Val Accuracy](assets/train_val_accuracy.png) |
|--------------------------------------------------|----------------------------------------------------------|

---

## 🚀 Deployment - Streamlit App

The model is deployed using **Streamlit** for real-time tumor prediction.



### 🛠️ How to Run the App

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

> 🖼️ *Upload an MRI image to receive a real-time classification.*

---

## 📁 Folder Structure

```
brain-tumor-classification/
│
├── data/
│   ├── train
│   ├── test
│   └── valid
│
├── models/
│   └── best_model.h5
│
├── notebooks/
│   └── training_experiments.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🤝 Contributors

- Aditya Singh 

---

## 📜 License

This project is licensed under the MIT License.

---

## 💬 Acknowledgements

- [MRI Brain Tumor Dataset](https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju)
- TensorFlow, Keras, Streamlit for ML model development and deployment.
