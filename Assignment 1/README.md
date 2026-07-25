# Fashion MNIST Image Classification using TensorFlow

## Overview

This assignment demonstrates how to build, train, and evaluate a simple neural network for classifying clothing images from the **Fashion MNIST** dataset using **TensorFlow** and **Keras**.

The model learns to recognize 10 different categories of fashion items and visualizes both the training process and prediction results.

---

## Features

* Loads the Fashion MNIST dataset
* Displays sample training images with labels
* Preprocesses image data by normalizing pixel values
* Builds a fully connected neural network using TensorFlow/Keras
* Trains the model with validation data
* Visualizes training and validation accuracy/loss
* Evaluates model performance on the test dataset
* Predicts clothing categories for test images
* Displays predictions with color-coded correctness

---

## Dataset

The assignment uses the **Fashion MNIST** dataset, which contains:

* **60,000** training images
* **10,000** testing images
* Image size: **28 × 28** grayscale pixels
* **10 clothing categories**

### Classes

| Label | Category    |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

---

## Technologies Used

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## Installation

Install the required packages:

```bash
pip install tensorflow matplotlib numpy
```

---

## Assignment Workflow

1. Load the Fashion MNIST dataset.
2. Display sample images with their labels.
3. Normalize pixel values to the range **0–1**.
4. Build a neural network consisting of:

   * Flatten layer
   * Dense layer (128 neurons, ReLU activation)
   * Dropout layer (20%)
   * Output layer (10 neurons, Softmax activation)
5. Compile the model using:

   * Optimizer: Adam
   * Loss Function: Sparse Categorical Crossentropy
   * Metric: Accuracy
6. Train the model for **10 epochs**.
7. Plot training and validation accuracy/loss.
8. Evaluate the model on the test dataset.
9. Predict labels for test images.
10. Display predictions alongside actual labels.

---

## Model Architecture

```text
Input (28×28)
      │
Flatten
      │
Dense (128, ReLU)
      │
Dropout (0.2)
      │
Dense (10, Softmax)
```

---

## Expected Output

The program will:

* Display sample training images
* Train the neural network
* Plot:

  * Training Accuracy
  * Validation Accuracy
  * Training Loss
  * Validation Loss
* Print the test accuracy, for example:

```text
Test Accuracy: 0.88
```

* Display predicted and actual labels for sample test images
* Print the prediction for the first test image, for example:

```text
Predicted: Sneaker
Actual: Sneaker
```

---

## File Structure

```text
Assignment 1/
│── Assignment1.ipynb
└── README.md
```

---

## Future Improvements

* Add Convolutional Neural Networks (CNNs) for improved accuracy.
* Save and reload the trained model.
* Add confusion matrix visualization.
* Implement data augmentation.
* Create a graphical user interface (GUI) or web application for image prediction.