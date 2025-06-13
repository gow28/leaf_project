
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gow28/leaf_project/blob/main/Project_original.ipynb)

📌 Project Title: Plant Disease Detection using Deep Learning with Grad-CAM visibility

🔍 Overview

This project introduces an AI-powered solution for **automated detection of plant leaf diseases** using deep learning. It harnesses the strength of **transfer learning with EfficientNetB0** to classify a diverse set of plant diseases accurately. Designed with modularity and extensibility in mind, the solution supports real-world applications such as farm advisory tools, plant health monitoring apps, and intelligent agricultural systems.

 🎯 Objectives

- To develop a deep learning model capable of classifying multiple plant diseases from leaf images.
- To employ a robust and lightweight architecture (EfficientNetB0) that ensures training efficiency and high accuracy.
- To support real-time predictions and scalable deployment options (e.g., API, mobile).
- To incorporate extensibility features like explainability and remedy generation.



🔑 Uniqueness & Innovation

Unlike many similar research or student projects, **this project stands out due to the following key differentiators**:

1. Automatic Model Reusability:  
   The system is designed to **detect and load a pre-trained model** seamlessly from cloud storage (Google Drive), saving time and compute costs. This allows the training process to be skipped unless required.

2. Real-Time Single Image Inference:  
   The model supports **manual prediction on individual leaf images**, enabling flexible testing and user interaction beyond bulk validation datasets.

3. Scalable Architecture for Integration:  
   The project is structured to be **API-ready**, making it easy to deploy as a RESTful service. This supports downstream integration into **web or mobile apps** for farmers or agronomists.

4. XAI-Ready Design (Explainable AI):  
   Planned support for **Grad-CAM visualization** helps interpret model decisions — a crucial feature for building trust in AI systems in agriculture.

5. Remedy Suggestion Pipeline:  
   A novel enhancement pipeline is proposed for future integration where the model can link each disease prediction to **structured treatment advice**, potentially sourced from a dynamic database or JSON file.

6. Professional Workflow & Automation:  
   The use of Google Colab + GitHub + Google Drive makes the project **collaborative and persistent**, minimizing retraining and allowing version control.

7. Compact Yet Accurate:  
   EfficientNetB0 is used instead of heavier architectures like ResNet50 or VGG, achieving **high accuracy (>95%)** while maintaining **faster training times** and smaller model size — ideal for **edge deployment**.

🗂️ Dataset

- Source: PlantVillage Dataset  
- Classes: 15 disease categories (including healthy leaf samples)  
- Images: 20,000+ RGB images  
- Target Crops: Tomato, Potato, Bell Pepper  
- Diseases Covered: Early blight, Late blight, Bacterial spot, Septoria leaf spot, Mold, Mites, and more



🧠 Model Architecture

- Base Model: EfficientNetB0 (pretrained on ImageNet)
- Head Layers:
  - GlobalAveragePooling2D
  - Dropout (0.5)
  - Dense Layer with L2 regularization
- Activation: Softmax (for multi-class output)
- Optimizer: Adam (LR = 1e-4)
- Loss: Categorical Crossentropy  
- Metrics: Accuracy, Precision, Recall

 🏋️ Training Configuration

- Input Size: 224x224 pixels  
- Batch Size: 32  
- Epochs: 15  
- Validation Split: 20%  
- Augmentations: Random rotation, zoom, shift, horizontal flip

✅ Key Features

- Automatically loads trained model from storage (avoids retraining)
- Full training history saved as JSON for transparency
- Classification report and accuracy metrics generated
- Single image prediction and probability visualization supported
- Visualization of training performance (accuracy/loss curves)

 📊 Results

- Validation Accuracy: Over 95%  
- Performance: Strong generalization across all 15 classes  
- Visual Plots: Training curves clearly show convergence and model stability

🔮 Future Scope

- Explainable AI Integration using Grad-CAM  
- Remedy Suggestion Engine using disease-remedy mappings (via JSON/CSV or database)  
- API Deployment with Flask or FastAPI for interaction via web/mobile interfaces  
- Edge & Mobile Optimization for real-time use by farmers in the field  
- Multilingual UI Support for local language accessibility  

🧾 Technologies Used

- Python 3.x  
- TensorFlow 2.18.0  
- Keras 3.8.0  
- NumPy, Matplotlib, Scikit-learn  
- Google Colab (Training & Development)  
- GitHub (Version Control & Hosting)

💡 Note
GitHub cannot display the full notebook because it’s >25MB.  
Use the **Colab badge above** ☝️ to view the complete notebook including outputs.
