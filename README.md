# Handwritten Digit Recognition using Convolutional Neural Networks (CNN) - PyTorch
This project aims to recognize handwritten digits using Convolutional Neural Networks (CNNs) implemented in PyTorch. The model is trained on the popular MNIST dataset, which consists of grayscale images of handwritten digits (0-9).

# Overview
The mnist_cnn_model.py script contains the code for defining, training, and testing the CNN model. This model architecture consists of two convolutional layers followed by max-pooling layers, and two fully connected layers for classification.

## Requirements

Ensure you have Python 3.x installed on your system along with the following dependencies:

- PyTorch
- torchvision
- matplotlib
- Pillow

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset
The MNIST dataset used for training and testing the model is automatically downloaded and stored in the data directory upon running the script for the first time.

## Model Architecture
The CNN model architecture used in this project is as follows:

- Convolutional Layer 1: 32 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: kernel size 2x2
- Convolutional Layer 2: 64 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: kernel size 2x2
- Fully Connected Layer 1: 64 * 7 * 7 input features, 128 output features, ReLU activation
- Fully Connected Layer 2: 128 input features, 10 output features (corresponding to digit classes)

## Training
The model is trained for 10 epochs using the Adam optimizer with a learning rate of 0.001 and Cross-Entropy Loss as the loss function. During training, the script prints the loss for each epoch.

## Testing
After training, the script evaluates the model's accuracy on the test set. The accuracy is calculated as the ratio of correctly predicted labels to the total number of labels.

## Prediction
You can use the predict.py script to make predictions on individual images. The predict_user_input function takes a trained model and the path to an image as input, and returns the predicted digit. Make sure to provide the path to the image you want to predict as image_path variable.

