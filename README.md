# Deep-Learning--Potato-Disease-Classification
Deep Learning- Potato Disease Classification
# Advanced Potato Disease Classification using Deep Learning and CNN

This repository implements a production-oriented deep learning system for automated potato leaf disease classification. The project uses a high-capacity Convolutional Neural Network built with TensorFlow and Keras to accurately identify disease patterns from high-resolution agricultural images, supporting early diagnosis and precision farming.

Overview

Plant diseases significantly impact agricultural productivity and food security. Manual disease identification is time-consuming, subjective, and not scalable. This project addresses these limitations by leveraging deep learning to automatically classify potato leaf diseases using image data.

The system classifies potato leaves into the following categories:

Potato___Early_blight
Potato___Late_blight
Potato___healthy

Key Features
High-resolution image classification using CNN
Robust multi-class prediction with confidence scoring
Reproducible training pipeline
Modular, deployment-ready codebase
Scalable design suitable for real-world agricultural AI systems

Dataset
Dataset Source: PlantVillage
Total Images: 2,152
Image Format: RGB
Number of Classes: 3
The dataset is structured into class-specific directories and validated through visualization and label verification prior to training.

Data Preprocessing Pipeline
Efficient data loading using image_dataset_from_directory
Image resizing to 256 by 256 pixels
Batch size of 32 for stable gradient optimization
Pixel normalization for faster convergence
Fixed random seed (seed = 123) to ensure reproducibility
This pipeline ensures consistent and memory-efficient training.

Model Architecture
The model is based on a Sequential Convolutional Neural Network designed to capture fine-grained spatial features related to plant diseases.
Architecture components include:
Convolutional layers for hierarchical feature extraction

Max pooling layers for spatial downsampling

Fully connected dense layers for classification

Softmax output layer for probabilistic predictions

Training Configuration

Input Shape: 256 x 256 x 3

Batch Size: 32

Epochs: 50

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metric: Accuracy

Extended training enables deeper feature learning and improved generalization.

Model Evaluation

The trained model demonstrates strong classification accuracy across all classes. Validation results confirm the model’s ability to generalize to unseen images and distinguish subtle disease-specific patterns.

Inference and Prediction

A custom inference pipeline is implemented for real-world usability. Given a raw leaf image, the system outputs:

Predicted disease class

Confidence score

This enables immediate validation and easy integration into downstream applications.

Deployment Readiness

The project is structured for seamless deployment:

Centralized configuration for image dimensions and class labels

Modular prediction logic

Compatible with FastAPI, Streamlit, and mobile or web applications

Practical Impact

Enables early disease detection

Reduces dependency on expert inspection

Supports scalable crop health monitoring

Contributes to sustainable and precision agriculture

Limitations

Performance depends on image quality and lighting conditions
Limited to potato leaf diseases in the current implementation
Field data may require additional fine-tuning

Future Enhancements
Transfer learning using EfficientNet, ResNet, or MobileNet
Expansion to multi-crop disease classification

Real-time inference using camera input

Cloud-based deployment for large-scale monitoring

Mobile application for farmer accessibility

Technologies Used

Python

TensorFlow

Keras

NumPy

Matplotlib

Repository Tags

Deep-Learning
Computer-Vision
TensorFlow
Keras
CNN
Agriculture-AI
Plant-Pathology
Crop-Disease-Detection
Image-Classification
