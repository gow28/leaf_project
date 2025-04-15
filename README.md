

📌 Project Title: Intelligent Plant Disease Detection using Deep Learning

 🔍 Overview

This project presents an AI-powered solution for the **automated identification of plant leaf diseases** using **deep learning techniques**. Leveraging the power of transfer learning through **EfficientNetB0**, the model is capable of accurately classifying multiple plant diseases from raw leaf images, providing a foundation for real-time agricultural diagnostics and support systems.

 🎯 Objectives

- To develop a deep learning-based model that accurately classifies plant leaf diseases.
- To utilize a pre-trained EfficientNetB0 architecture for efficient and scalable training.
- To enable predictions on unseen leaf images for real-world applicability.
- To ensure the model is reusable, extensible, and deployable for practical usage.



 🗂️ Dataset

- **Source**: PlantVillage Dataset  
- **Data Type**: RGB Leaf Images  
- **Number of Classes**: 15 disease categories (including healthy samples)  
- **Size**: ~20,000+ images  

The dataset encompasses various diseases affecting crops like **Tomato, Potato, and Bell Pepper**, including early blight, bacterial spots, mold, and viral infections.



🧠 Model Architecture

- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Customization**:
  - Global Average Pooling
  - Dropout Layer (0.5)
  - Dense Layer with Softmax Activation for Multi-Class Classification

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report  



 🏋️ Training Configuration

- **Image Size**: 224 x 224 pixels  
- **Batch Size**: 32  
- **Epochs**: 15  
- **Validation Split**: 20%  
- **Data Augmentation**: Rotation, Zoom, Shifting, Horizontal Flip  

Training was conducted using TensorFlow and Keras in a Google Colab environment. Model persistence and training history were handled using Google Drive to avoid retraining on every session.


 ✅ Key Features

- **Model Auto-loading**: Automatically loads the pre-trained model if available, otherwise trains from scratch.
- **Performance Evaluation**: Generates a comprehensive classification report with accuracy scores.
- **Image Prediction**: Provides prediction functionality for individual test images with softmax probabilities.
- **Visualization**: Plots training history (accuracy and loss) for transparent model behavior monitoring.



📊 Results

The model demonstrated **high classification accuracy (>95%)** on the validation set, proving its capability to generalize well across various plant leaf disease categories.


 🔮 Future Scope

- **Explainable AI**: Integrate Grad-CAM to visualize affected regions on leaves, enhancing model interpretability.
- **Remedy Integration**: Connect predictions with treatment recommendations using structured external sources (e.g., JSON/CSV).
- **API Deployment**: Wrap the model into a RESTful API using Flask or FastAPI, enabling web or mobile applications to interact with the AI model for real-time diagnostics.
- **Edge Deployment**: Optimize the model for deployment on mobile devices or edge hardware for in-field use by farmers.
- **Multilingual UI**: Add multi-language support to make the solution accessible to non-English speaking agricultural communities.



 🧾 Technologies Used

- Python 3.x  
- TensorFlow 2.18.0  
- Keras 3.8.0  
- NumPy, Matplotlib, Scikit-learn  
- Google Colab (for development and training)

