# Assignment 2 – Artificial Neural Network (ANN) on Iris Dataset

## Overview

This assignment demonstrates the implementation of an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** for multiclass classification on the **Iris dataset**. The workflow includes data preprocessing, feature normalization, model training, prediction, and performance evaluation.

## Objective

- Load the Iris dataset.
- Split the dataset into training and testing sets.
- Standardize the input features.
- Build and train an ANN using TensorFlow/Keras.
- Evaluate the model using accuracy, confusion matrix, and classification report.

## Dataset

The **Iris dataset** is a classic multiclass classification dataset available in Scikit-learn.

- **Total Samples:** 150
- **Features:** 4
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Classes:** 3
  - Setosa
  - Versicolor
  - Virginica

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

## Model Architecture

| Layer | Neurons | Activation |
|--------|--------:|------------|
| Input Layer | 4 | - |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 3 | Softmax |

## Project Workflow

1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Standardize the input features using `StandardScaler`.
4. Build a Sequential ANN model.
5. Compile the model using the Adam optimizer.
6. Train the model for 100 epochs.
7. Predict the test data classes.
8. Evaluate the model using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

## Running the assignment

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
Accuracy: 0.9667
```

Example Confusion Matrix:

```
[[10 0 0]
 [0 10 0]
 [0 1 9]]
```

A detailed **Classification Report** containing Precision, Recall, F1-Score, and Support is also generated.

## Learning Outcomes

- Understanding Artificial Neural Networks (ANNs)
- Data preprocessing and feature scaling
- Building neural networks using TensorFlow/Keras
- Multiclass classification
- Model evaluation using standard performance metrics

## Files

```
Assignment-2/
├── assignment2.ipynb
└── README.md
```

## Author

**Nitish Charanlal Sahu** 

B.Tech – Computer Science & Engineering (Artificial Intelligence)

Vishwakarma Institute of Technology, Pune

## License

This project is submitted as part of the **Deep Learning Laboratory** coursework for Semester V at Vishwakarma Institute of Technology, Pune.