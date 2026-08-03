# Assignment 3 – Forward Propagation and Backpropagation using TensorFlow/Keras

## Overview

This assignment demonstrates the implementation of **Forward Propagation** and **Backpropagation** using **TensorFlow/Keras** on the **CIFAR-10** dataset. The project analyzes the effect of different **learning rates** and **number of epochs** on the performance of an Artificial Neural Network (ANN).

## Objective

- Load and preprocess the CIFAR-10 dataset.
- Build an Artificial Neural Network (ANN) using TensorFlow/Keras.
- Train the model using forward propagation and backpropagation.
- Analyze the effect of different learning rates on model performance.
- Analyze the effect of varying the number of epochs.
- Evaluate the trained model using test accuracy and loss.

## Dataset

The **CIFAR-10** dataset is a benchmark image classification dataset available in TensorFlow/Keras.

- **Training Samples:** 50,000
- **Testing Samples:** 10,000
- **Image Size:** 32 × 32 × 3 (RGB)
- **Classes:** 10

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Model Architecture

| Layer | Configuration | Activation |
|--------|--------------|------------|
| Input Layer | 32 × 32 × 3 | - |
| Flatten Layer | 3072 Units | - |
| Hidden Layer 1 | 256 Neurons | ReLU |
| Hidden Layer 2 | 128 Neurons | ReLU |
| Output Layer | 10 Neurons | Softmax |

## Assignment Workflow

1. Load the CIFAR-10 dataset.
2. Normalize the image pixel values to the range **0–1**.
3. Build a Sequential Artificial Neural Network.
4. Compile the model using the Adam optimizer.
5. Train the model using forward propagation and backpropagation.
6. Evaluate the model on the test dataset.
7. Compare validation accuracy using different learning rates.
8. Compare validation accuracy using different numbers of epochs.
9. Visualize the results using accuracy graphs.
10. Analyze the impact of learning rate and epochs on model performance.

## Running the Assignment

```cmd
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Learning Rate Analysis

The following learning rates were used during experimentation:

- **0.1**
- **0.01**
- **0.001**
- **0.0001**

A validation accuracy graph is generated to compare the effect of each learning rate on model performance.

## Epoch Analysis

The model was trained using different numbers of epochs:

- **5**
- **10**
- **20**
- **30**

A validation accuracy graph is generated to analyze the effect of increasing the number of training epochs.

## Sample Output

```
Learning Rate : 0.001
Epochs        : 10
Test Accuracy : 0.51
Test Loss     : 1.38
```

*(Accuracy and loss values may vary slightly with each execution.)*

The assignment also generates:

- Validation Accuracy vs Learning Rate graph
- Validation Accuracy vs Number of Epochs graph

## Observation

- A learning rate of **0.001** produced the most stable training.
- Very high learning rates resulted in unstable convergence.
- Very low learning rates caused slower learning.
- Increasing the number of epochs generally improved performance until the model converged.

## Why is Accuracy Only Around 50%?

The implemented model consists of:

- Flatten Layer
- Dense Layer (256 neurons)
- Dense Layer (128 neurons)
- Dense Layer (10 neurons with Softmax activation)

This is a **fully connected Artificial Neural Network (Multi-Layer Perceptron - MLP)**.

The CIFAR-10 dataset contains images with important **spatial patterns** such as edges, textures, and object shapes. Using a **Flatten** layer removes this spatial information before it reaches the dense layers, making it difficult for the network to learn meaningful image features.

As a result, the model achieves an accuracy of only **around 50%**.

**Convolutional Neural Networks (CNNs)** are specifically designed for image classification because they preserve spatial information through convolutional layers. Replacing the MLP with a CNN would significantly improve the model's accuracy on the CIFAR-10 dataset.

## Learning Outcomes

- Understanding Forward Propagation
- Understanding Backpropagation
- Building Artificial Neural Networks using TensorFlow/Keras
- Image preprocessing and normalization
- Effect of learning rate on neural network training
- Effect of epochs on model convergence
- Model evaluation using accuracy and loss

## Files

```
Assignment-3/
├── assignment3.ipynb
└── README.md
```

## Author

**Nitish Charanlal Sahu**

B.Tech – Computer Science & Engineering (Artificial Intelligence)

Vishwakarma Institute of Technology, Pune

## License

This project is submitted as part of the **Deep Learning Laboratory** coursework for Semester V at Vishwakarma Institute of Technology, Pune.