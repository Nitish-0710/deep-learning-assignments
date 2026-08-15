# Assignment 4 – Convolutional Neural Network for Tomato Disease Classification

## 📌 Aim

To design and implement a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** for classifying tomato leaf images into different disease and healthy categories using the **PlantVillage dataset**.


## 🎯 Objectives

- Understand the application of CNNs for image classification.
- Load and preprocess plant leaf images using TensorFlow.
- Select tomato disease classes from the PlantVillage dataset.
- Create training, validation, and independent test datasets.
- Apply image normalization and data augmentation.
- Handle class imbalance using class weights.
- Design and train a CNN using TensorFlow/Keras.
- Evaluate the trained model using accuracy, loss, precision, recall, F1-score, and a confusion matrix.
- Analyze prediction confidence and per-class performance.
- Save and reload the trained model for reproducibility.


## 📊 Dataset

The **PlantVillage dataset** is used for tomato leaf disease classification.

### Dataset Source

The dataset was downloaded from Kaggle:

**PlantVillage Dataset:**  
https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

The downloaded dataset contains multiple crops and disease categories. The `train` directory contains **40 class directories**, including classes belonging to Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

For this assignment, only the **10 Tomato classes** are selected programmatically.

### Original Dataset Structure

```text
PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   ├── Blueberry___healthy/
│   ├── Cherry_(including_sour)___healthy/
│   ├── Cherry_(including_sour)___Powdery_mildew/
│   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
│   ├── Corn_(maize)___Common_rust_/
│   ├── Corn_(maize)___healthy/
│   ├── Corn_(maize)___Northern_Leaf_Blight/
│   ├── Grape___Black_rot/
│   ├── Grape___Esca_(Black_Measles)/
│   ├── Grape___healthy/
│   ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
│   ├── Orange___Haunglongbing_(Citrus_greening)/
│   ├── Peach___Bacterial_spot/
│   ├── Peach___healthy/
│   ├── Pepper,_bell___Bacterial_spot/
│   ├── Pepper,_bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___healthy/
│   ├── Potato___Late_blight/
│   ├── Raspberry___healthy/
│   ├── Soybean___healthy/
│   ├── Squash___Powdery_mildew/
│   ├── Strawberry___healthy/
│   ├── Strawberry___Leaf_scorch/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___healthy/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│
└── val/
    └── ... same class structure ...
```

The notebook identifies the tomato classes by selecting directories whose names begin with:

```text
Tomato___
```

## 🍅 Tomato Classes Used

A total of **10 tomato classes** are used for the classification task.

| Class | Disease |
|------:|---------|
| 0 | Bacterial spot |
| 1 | Early blight |
| 2 | Late blight |
| 3 | Leaf Mold |
| 4 | Septoria leaf spot |
| 5 | Spider mites / Two-spotted spider mite |
| 6 | Target Spot |
| 7 | Tomato Yellow Leaf Curl Virus |
| 8 | Tomato mosaic virus |
| 9 | Healthy |


## 🔀 Dataset Split

The original `train` directory contains **14,529 tomato images**.

A **stratified 85:15 split** is performed on these images:

- 85% → Training
- 15% → Validation

The original `val` directory is kept separate and is used as an **independent test dataset**.

| Dataset | Images |
|---|---:|
| Training | 12,349 |
| Validation | 2,180 |
| Independent Testing | 3,631 |
| **Total** | **18,160** |

### Dataset Organization

```text
PlantVillage
│
├── train/
│   │
│   └── Tomato classes
│          │
│          └── 14,529 images
│                   │
│             Stratified 85:15 split
│                /           \
│               /             \
│      12,349 Training    2,180 Validation
│
└── val/
    │
    └── Tomato classes
           │
           └── 3,631 Independent Test Images
```


## 🛠️ Technologies Used

- **Python 3.13**
- **TensorFlow 2.21.0**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**


## 🧹 Image Preprocessing

All images are preprocessed before being passed to the CNN.

### Configuration

```text
Image Height : 128
Image Width  : 128
Channels     : 3
Batch Size   : 32
```

The following preprocessing operations are performed:

1. Images are loaded from their respective directories.
2. Images are decoded as RGB images.
3. Images are resized to `128 × 128`.
4. Pixel values are normalized to the range `0–1`.
5. Images are grouped into batches of 32.
6. TensorFlow `tf.data` pipelines are used for efficient data loading and prefetching.

## 🔄 Data Augmentation

Data augmentation is applied to the training images to improve model generalization.

The following augmentation techniques are used:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast adjustment

```text
RandomFlip
RandomRotation
RandomZoom
RandomContrast
```

Validation and test images are not augmented.

## ⚖️ Class Imbalance Handling

The dataset contains an unequal number of images across the 10 tomato disease classes.

For example, **Tomato Yellow Leaf Curl Virus** has significantly more training images than **Tomato mosaic virus**.

To reduce the effect of class imbalance, class weights are calculated from the training dataset.

The calculated class weights are passed to the training process so that classes with fewer images receive greater importance during optimization.

### Training Class Distribution

| Class | Disease | Training Images |
|------:|---------|----------------:|
| 0 | Bacterial spot | 1,447 |
| 1 | Early blight | 680 |
| 2 | Late blight | 1,298 |
| 3 | Leaf Mold | 647 |
| 4 | Septoria leaf spot | 1,204 |
| 5 | Spider mites / Two-spotted spider mite | 1,140 |
| 6 | Target Spot | 954 |
| 7 | Tomato Yellow Leaf Curl Virus | 3,643 |
| 8 | Tomato mosaic virus | 254 |
| 9 | Healthy | 1,082 |


## 🧠 CNN Architecture

A custom Convolutional Neural Network is implemented using TensorFlow/Keras.

### Architecture

```text
Input Image
     ↓
Data Augmentation
     ↓
Conv2D - 32 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 64 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 128 Filters
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense - 128 Neurons
     ↓
Dropout
     ↓
Dense - 10 Neurons
     ↓
Softmax Output
```

### Model Parameters

```text
Total Parameters     : 4,288,970
Trainable Parameters : 4,288,970
Non-trainable        : 0
```

## ⚙️ Model Compilation

The CNN is compiled using the following configuration:

| Configuration | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Metric | Accuracy |


## 🏋️ Model Training

The CNN was trained for a maximum of **30 epochs**.

Two callbacks were used:

### EarlyStopping

Early stopping terminates training when the validation loss stops improving and restores the weights from the best-performing epoch.

### ReduceLROnPlateau

The learning rate is reduced when the validation loss stops improving.

### Training Configuration

```text
Maximum Epochs : 30
Batch Size     : 32
Initial LR     : 0.001
Optimizer      : Adam
```

The training stopped early at **epoch 27**, and the model restored the weights from **epoch 22**, which produced the best validation performance.


## 📈 Training Results

The best validation performance was obtained at **Epoch 22**.

| Metric | Result |
|---|---:|
| Best Epoch | 22 |
| Best Validation Accuracy | **94.95%** |
| Best Validation Loss | **0.1839** |

The notebook generates training history plots for:

- Training vs Validation Accuracy
- Training vs Validation Loss

These plots help analyze model learning and generalization behavior.

## 🧪 Final Test Evaluation

The final trained CNN was evaluated on the **3,631 independent test images**.

### Final Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **95.79%** |
| Test Loss | **0.1362** |
| Macro Precision | **94.00%** |
| Macro Recall | **95.77%** |
| Macro F1-Score | **94.76%** |
| Weighted F1-Score | **95.82%** |

### Prediction Results

```text
Total Predictions       : 3,631
Correct Predictions     : 3,478
Incorrect Predictions   : 153
Correct Prediction Rate : 95.79%
Average Confidence      : 96.24%
```


## 📋 Classification Report

The following table summarizes the class-wise performance of the trained CNN.

| Disease Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Bacterial spot | 99.49% | 92.24% | 95.73% | 425 |
| Early blight | 87.16% | 95.00% | 90.91% | 200 |
| Late blight | 97.50% | 91.88% | 94.61% | 382 |
| Leaf Mold | 95.88% | 97.38% | 96.62% | 191 |
| Septoria leaf spot | 96.88% | 96.61% | 96.75% | 354 |
| Spider mites / Two-spotted spider mite | 93.81% | 90.45% | 92.10% | 335 |
| Target Spot | 86.58% | 96.44% | 91.25% | 281 |
| Tomato Yellow Leaf Curl Virus | 98.32% | 98.32% | 98.32% | 1,071 |
| Tomato mosaic virus | 85.06% | 100.00% | 91.93% | 74 |
| Healthy | 99.37% | 99.37% | 99.37% | 318 |

### Overall Metrics

| Metric | Score |
|---|---:|
| Accuracy | **95.79%** |
| Macro Precision | **94.00%** |
| Macro Recall | **95.77%** |
| Macro F1-Score | **94.76%** |
| Weighted F1-Score | **95.82%** |

## 🔍 Confusion Matrix

A confusion matrix is generated to analyze correct and incorrect predictions for all 10 tomato classes.

The model performs particularly well on:

- Tomato mosaic virus
- Healthy
- Tomato Yellow Leaf Curl Virus
- Leaf Mold
- Septoria leaf spot

The major classification errors occur among visually similar disease categories, particularly:

- Bacterial spot
- Spider mites
- Late blight
- Early blight
- Target Spot

The confusion matrix is included in the notebook as part of the model evaluation.

## 📊 Per-Class Accuracy

The model achieved the following class-wise accuracies:

| Disease Class | Accuracy |
|---|---:|
| Bacterial spot | 92.24% |
| Spider mites / Two-spotted spider mite | 90.45% |
| Late blight | 91.88% |
| Tomato Yellow Leaf Curl Virus | 98.32% |
| Septoria leaf spot | 96.61% |
| Early blight | 95.00% |
| Target Spot | 96.44% |
| Leaf Mold | 97.38% |
| Healthy | 99.37% |
| Tomato mosaic virus | 100.00% |

The model achieves **90% or higher accuracy for every class**.

## 🎯 Prediction Confidence Analysis

Prediction confidence was analyzed for the complete independent test dataset.

| Metric | Result |
|---|---:|
| Overall Average Confidence | **96.24%** |
| Average Confidence of Correct Predictions | **97.21%** |
| Average Confidence of Incorrect Predictions | **74.30%** |

The lower average confidence for incorrect predictions indicates that the model was generally less confident when making classification errors.


## 💾 Model Saving and Verification

The trained CNN model is saved in Keras format:

```text
tomato_disease_cnn.keras
```

### Model Size

```text
49.14 MB
```

The saved model was loaded again using TensorFlow/Keras and evaluated on the independent test dataset.

### Verification

```text
Original Model Test Accuracy : 95.79%
Loaded Model Test Accuracy   : 95.79%
Accuracy Difference          : 0.000000
```

The zero accuracy difference confirms that the saved model preserves the trained architecture and learned weights correctly.


## 📁 Repository Structure

```text
Assignment 4/
│
├── Assignment4.ipynb
├── README.md
├── tomato_disease_cnn.keras
│
└── PlantVillage/              # Excluded from Git
    ├── train/                 # Original 38-class training directory
    └── val/                   # Original validation directory
```

The following files/directories are intentionally excluded from Git:

```text
PlantVillage/
*.zip
```

The dataset is not included in the repository because of its large size.


## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Nitish-0710/deep-learning-assignments
cd deep-learning-assignments
```

### 2. Install Required Dependencies



```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the PlantVillage dataset from Kaggle:

https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Extract the dataset and place the `PlantVillage` directory inside the Assignment 4 directory:

```text
Assignment 4/
├── Assignment4.ipynb
├── README.md
└── PlantVillage/
    ├── train/
    └── val/
```

### 4. Open the Notebook in Jupyter or VS Code

```text
Assignment4.ipynb
```

### 5. Run the Notebook

Run the notebook cells sequentially.

The notebook performs the following workflow:

```text
Load Dataset
      ↓
Identify Tomato Classes
      ↓
Analyze Dataset
      ↓
Create Stratified Train/Validation Split
      ↓
Create Independent Test Set
      ↓
Preprocess Images
      ↓
Create TensorFlow Data Pipelines
      ↓
Apply Data Augmentation
      ↓
Build CNN Model
      ↓
Calculate Class Weights
      ↓
Compile Model
      ↓
Train CNN
      ↓
Evaluate Model
      ↓
Generate Predictions
      ↓
Confusion Matrix
      ↓
Classification Report
      ↓
Prediction Confidence Analysis
      ↓
Save Model
      ↓
Reload and Verify Model
```

## 📌 Final Results Summary

```text
Dataset                  : PlantVillage
Selected Classes         : 10
Training Images          : 12,349
Validation Images        : 2,180
Independent Test Images  : 3,631

Image Size               : 128 × 128 × 3
Batch Size               : 32

Optimizer                : Adam
Initial Learning Rate    : 0.001
Maximum Epochs           : 30
Best Epoch               : 22

Best Validation Accuracy : 94.95%
Best Validation Loss     : 0.1839

Test Accuracy            : 95.79%
Test Loss                : 0.1362

Macro Precision          : 94.00%
Macro Recall             : 95.77%
Macro F1-Score            : 94.76%
Weighted F1-Score         : 95.82%

Correct Predictions      : 3,478
Incorrect Predictions    : 153
Average Confidence       : 96.24%

Saved Model              : tomato_disease_cnn.keras
Model Size               : 49.14 MB
```


## 🏁 Conclusion

A Convolutional Neural Network was successfully designed and implemented using TensorFlow/Keras for multi-class tomato disease classification using the PlantVillage dataset.

The experiment used **10 tomato disease and healthy classes** with **12,349 training images**, **2,180 validation images**, and **3,631 independent test images**.

Image preprocessing, data augmentation, and class weighting were used to improve model generalization and handle class imbalance.

The trained CNN achieved an **independent test accuracy of 95.79%** and a **macro F1-score of 94.76%**, demonstrating strong classification performance across the selected tomato classes.

The model was also saved and reloaded successfully, producing an **accuracy difference of 0.000000** between the original and reloaded models. This confirms that the trained model can be stored and reused reliably.

Overall, the experiment demonstrates the effectiveness of CNNs for automated tomato leaf disease classification using image-based deep learning.

---