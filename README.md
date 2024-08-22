
# Potato Disease Prediction

This project involves building a convolutional neural network (CNN) model using TensorFlow to classify images of potato leaves into different disease categories. The workflow includes data loading, preprocessing, model creation, training, evaluation, and prediction. The code is designed to be run in a Jupyter Notebook environment.

## Table of Contents
1. **Project Overview**
2. **Installation**
3. **Workflow**
4. **Usage**
5. **Model Evaluation**
6. **Saving and Loading the Model**

## Project Overview

The primary objective of this project is to accurately classify potato leaf images into different categories of disease or healthy leaves. The dataset is processed, and a CNN model is trained to achieve high accuracy in prediction.

## Installation

To run this project, you'll need the following libraries installed in your Python environment: TensorFlow, Matplotlib, and Scikit-learn. You can install these libraries using pip.

## Workflow

### 1. **Data Loading**

The dataset is loaded using TensorFlow's `image_dataset_from_directory` method. The images are organized in batches of size 32 with a resolution of 256x256 pixels.

### 2. **Data Visualization**

A few images from the dataset are visualized along with their respective labels to get an understanding of the data.

### 3. **Data Splitting**

The dataset is split into training, validation, and test datasets. **80%** of the data is used for training, and the remaining **20%** is split equally between validation and testing.

### 4. **Data Preprocessing**

The images are resized and rescaled to standardize the input to the CNN. Additionally, data augmentation techniques are applied to make the model more robust.

### 5. **Model Creation**

A CNN model is constructed with multiple convolutional layers followed by max-pooling layers and a fully connected dense layer. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

### 6. **Model Training**

The model is trained for 30 epochs using the training dataset, and its performance is validated using the validation dataset.

## Usage

### **Prediction**

To predict the class of a new image, the model processes the image and outputs the predicted class along with the confidence level.

### **Visualization of Predictions**

The model’s predictions are visualized with the actual class labels and confidence scores for each prediction.

## Model Evaluation

The trained model is evaluated on the test dataset to determine its accuracy and loss. The model outputs the **loss** and **accuracy** on the test dataset, which helps in assessing its performance.

## Saving and Loading the Model

After training, the model is saved for later use. The saved model can be loaded for making predictions without needing to retrain it.
