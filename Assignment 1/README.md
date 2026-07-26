# Assignment 1 – Fashion MNIST Image Classification using TensorFlow

## Overview

This assignment demonstrates the implementation of a simple **Artificial Neural Network (ANN)** using **TensorFlow/Keras** for image classification on the **Fashion MNIST** dataset. The workflow includes data preprocessing, normalization, model training, prediction, visualization, and performance evaluation.

## Objective

- Load the Fashion MNIST dataset.
- Visualize sample images and their corresponding labels.
- Normalize image pixel values.
- Build and train an Artificial Neural Network using TensorFlow/Keras.
- Evaluate the model using test accuracy.
- Visualize the training process and prediction results.

## Dataset

The **Fashion MNIST** dataset is a benchmark image classification dataset available in TensorFlow/Keras.

- **Training Samples:** 60,000
- **Testing Samples:** 10,000
- **Image Size:** 28 × 28 grayscale pixels
- **Classes:** 10

### Class Labels

| Label | Category |
|------:|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Model Architecture

| Layer | Configuration | Activation |
|--------|--------------|------------|
| Input Layer | 28 × 28 | - |
| Flatten Layer | 784 Units | - |
| Hidden Layer | 128 Neurons | ReLU |
| Dropout Layer | 20% | - |
| Output Layer | 10 Neurons | Softmax |

## Assignment Workflow

1. Load the Fashion MNIST dataset.
2. Display sample training images with their labels.
3. Normalize pixel values to the range **0–1**.
4. Build a Sequential Artificial Neural Network.
5. Compile the model using the Adam optimizer.
6. Train the model for **10 epochs** with validation data.
7. Plot the training and validation accuracy.
8. Plot the training and validation loss.
9. Evaluate the model on the test dataset.
10. Predict class labels for the test images and visualize the results.

## Running the Assignment

```cmd
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Sample Output

```
Test Accuracy: 0.90
```

The program also generates:

- Sample Fashion MNIST training images
- Training Accuracy vs Validation Accuracy graph
- Training Loss vs Validation Loss graph
- Predictions on test images with correct predictions shown in **green** and incorrect predictions shown in **red**

Example Prediction:

```
Predicted: Sneaker
Actual: Sneaker
```

## Learning Outcomes

- Understanding image classification using Artificial Neural Networks (ANNs)
- Image preprocessing and normalization
- Building neural networks using TensorFlow/Keras
- Training and evaluating deep learning models
- Visualizing model performance using accuracy and loss graphs

## Files

```
Assignment-1/
├── assignment1.ipynb
└── README.md
```

## Author

**Nitish Charanlal Sahu**

B.Tech – Computer Science & Engineering (Artificial Intelligence)

Vishwakarma Institute of Technology, Pune

## License

This project is submitted as part of the **Deep Learning Laboratory** coursework for Semester V at Vishwakarma Institute of Technology, Pune.