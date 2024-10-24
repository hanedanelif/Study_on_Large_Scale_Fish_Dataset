# Study_on_Large_Scale_Fish_Dataset

https://www.kaggle.com/code/elifhanedan/study-on-large-scale-fish-dataset

Fish Species Classification Using Deep Learning
This project aims to classify various species of fish using deep learning techniques. The dataset used is A Large Scale Fish Dataset from Kaggle, containing images of different fish species. We apply data preprocessing, image augmentation, and deep learning model training to achieve optimal classification results.

# Project Overview
In this project, we utilize a convolutional neural network (CNN) built with TensorFlow/Keras to classify fish species. The dataset consists of multiple fish classes, and the goal is to build a model that can correctly classify the images into their respective species.

# Key Steps:
1-Data Preprocessing:

Resizing images to 225x225 pixels.
Normalizing pixel values.
Splitting the dataset into training, validation, and test sets (80/10/10 ratio).

2-Image Augmentation:

Applied techniques like rotation, shifting, zoom, brightness adjustment, and horizontal flipping to enhance the model's robustness.

3-Model Architecture:

Sequential model with several fully connected dense layers.
Dropout layers to prevent overfitting.
Softmax activation for multiclass classification.

4-Training:

Adam optimizer with a low learning rate for better convergence.
Early stopping mechanism to halt training when the validation loss stops improving.

5-Evaluation:

Accuracy and loss metrics on the test set.
Confusion matrix and classification report to assess the model's performance on individual classes.

# Dataset
The dataset used in this project is sourced from Kaggle and contains images of multiple fish species. Each image is labeled with its respective species. The dataset has been preprocessed by resizing and normalizing the images for input into the neural network.


# Model Performance
Test Accuracy: 57.9%
The model shows varying performance across different classes. Certain fish species are classified with high accuracy, while others are challenging for the model, indicating the need for further improvement.

# Confusion Matrix and Classification Report
The confusion matrix and classification report provide deeper insights into the model’s performance by showing metrics like precision, recall, and F1-score for each fish class. Some species are classified with high precision and recall, while others require improvement.

# Future Improvements
Fine-tuning the model architecture (e.g., adding convolutional layers).
Using transfer learning to leverage pre-trained models like ResNet or VGG16.
Improving data augmentation techniques to handle class imbalance.

# Conclusion
This project serves as an introduction to deep learning for image classification. While the model performs moderately well, there is room for improvement, particularly in handling difficult classes. By incorporating more advanced models and techniques, classification accuracy can be further enhanced.
