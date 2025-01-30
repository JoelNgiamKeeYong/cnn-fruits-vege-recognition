# 🍎 Fruits & Vegetables Recognition System using CNN

## 🚀 Business Scenario

Food classification using computer vision has many practical applications, from automated checkout systems to dietary tracking and food waste reduction. This project focuses on developing a **Fruits & Vegetables Recognition System** that can accurately classify images of fruits and vegetables using a **Convolutional Neural Network (CNN)** trained on a **Kaggle dataset**. A user-friendly interface is deployed for seamless interaction with the model, utilizing Streamlit for frontend development and Heroku for hosting the application.

---

## 🧠 Business Problem

Recognizing different fruits and vegetables in images is a challenging task due to variations in shape, color, and texture. The goal is to build an **image classification model** that can accurately predict the type of fruit or vegetable from an image, enabling applications in supermarkets, smart kitchens, and health tracking systems.

---

## 🛠️ Solution Approach

This project follows a **deep learning workflow** with the following steps:

### 1️⃣ **Data Collection and Preprocessing**

- **Dataset**:
  - Used the **Fruits and Vegetables Image Recognition Dataset** from Kaggle ([Dataset Link](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)).
  - The data is not included in this directory due to file size constraints. To see the full dataset, see the Kaggle website link above.
- **Image Preprocessing**:
  - Resized images to **64x64 pixels** for uniform input size.
  - Applied **normalization** to scale pixel values between 0 and 1.

### 2️⃣ **Model Building (CNN Architecture)**

- **Convolutional Layers**: Extracted spatial features from images.
- **Max-Pooling Layers**: Reduced dimensionality while preserving key information.
- **Flattening Layer**: Converted feature maps into a 1D array.
- **Fully Connected Layers (Dense Layers)**: Made final predictions using softmax activation.
- **Activation Functions**: Used **ReLU** for hidden layers and **softmax** for classification.

### 3️⃣ **Model Training and Evaluation**

- **Loss Function**: Categorical Crossentropy (for multi-class classification).
- **Optimizer**: Adam optimizer for efficient learning.
- **Evaluation Metrics**:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Visualizations**: Created visualizations to understand model performance across validation and test sets.

---

## 📊 Model Performance

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 95.2% |
| Validation Accuracy | 91.9% |
| Loss (Validation)   | 0.24  |

---

## ⚠️ Limitations

1️⃣ **Dataset Constraints**

- The dataset may not include all possible lighting conditions, backgrounds, or rare fruit varieties.
- Performance may decrease when tested on images outside the training distribution.

2️⃣ **Model Generalization**

- The CNN model performs well on the provided dataset but may require fine-tuning for real-world applications with unseen data.

3️⃣ **Real-World Deployment**

- For real-time classification, additional optimizations like **quantization** or **TensorFlow Lite conversion** would be needed for deployment on edge devices.

---

## 🧠 Key Skills Demonstrated

✅ **Deep Learning with CNNs**  
✅ **Image Preprocessing & Augmentation**  
✅ **Model Evaluation & Performance Metrics**  
✅ **TensorFlow/Keras for Model Training**  
✅ **Data Handling using Pandas & NumPy**  
✅ **Visualization with Matplotlib & Seaborn**  
✅ **Streamlit for Frontend Development**  
✅ **Heroku Hosting for Deployment**

---

## 🛠️ Technical Tools & Libraries

- **Python**: Main programming language.
- **TensorFlow/Keras**: For deep learning model development.
- **OpenCV**: For image preprocessing.
- **Matplotlib**: For data visualization.
- **NumPy**: For data handling and analysis.
- **Streamlit**: For creating an interactive web frontend to visualize the model’s predictions.
- **Heroku**: For hosting the application online, making it accessible to users.
