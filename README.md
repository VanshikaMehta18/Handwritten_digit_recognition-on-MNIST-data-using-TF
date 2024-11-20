# Handwritten_digit_recognition-on-MNIST-data-using-TF

Hi,
This is a machine learning project aimed at recognizing handwritten digits using TensorFlow and the MNIST dataset. 
The project leverages Convolutional Neural Networks (CNNs), a deep learning architecture well-suited for image classification tasks, to classify 28x28 grayscale images of digits (0-9).
The goal of this project is to accurately predict the digit in an image from the MNIST dataset, achieving a high level of accuracy using deep learning techniques.

**Model Architecture**
Flatten Layer: Converts the 28x28 pixel input images into a one-dimensional array of 784 values.
Dense Layers: Two fully connected hidden layers, each with 128 neurons and ReLU activation.
Output Layer: The final layer has 10 neurons (one for each digit 0-9), with a softmax activation function to output a probability distribution over the 10 classes.
Optimizer: Adam optimizer for efficient gradient descent.
Loss Function: Sparse categorical crossentropy because the labels are integers (not one-hot encoded).
Metrics: Accuracy to track the model’s performance.

**Performance**
The model achieves 97%+ accuracy on the MNIST test dataset. The performance can be further improved with hyperparameter tuning or by experimenting with more advanced architectures.

