# Deep Learning Assignment 7 — RNN, LSTM and GRU for Sequence Classification

## 📌 Problem Statement

**Implement and compare RNN, LSTM, and GRU models for sequence classification, and analyze their performance using appropriate evaluation metrics.**

This assignment implements and compares three recurrent neural network architectures — **Simple RNN, LSTM, and GRU** — for binary sentiment classification using the **IMDB Movie Reviews dataset**.

All three models use the same preprocessing pipeline and comparable architecture settings so that their performance can be evaluated fairly.

---

## 🎯 Objectives

- Implement a **Simple RNN** model, **LSTM**, **GRU** for sequence classification.
- Train all three models using a consistent experimental setup.
- Compare their performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

---

## 📂 Dataset

### IMDB Movie Reviews

The **IMDB Movie Reviews dataset** is a binary sentiment classification dataset containing movie reviews labeled as:

- `0` → Negative
- `1` → Positive

The dataset contains:

| Dataset | Samples |
|---|---:|
| Training data | 25,000 |
| Test data | 25,000 |
| Total | 50,000 |

The dataset is loaded using:

```python
from tensorflow.keras.datasets import imdb
```

The original training set is further divided into:

* **20,000 samples** → Training
* **5,000 samples** → Validation

The official **25,000-sample test set remains untouched** until final evaluation.

---

## 🧠 Model Architecture

All three models use the same overall architecture:

```text
Input Sequence
      ↓
Embedding Layer
      ↓
Recurrent Layer
      ↓
Dropout
      ↓
Dense Output Layer
      ↓
Sigmoid
```

### Common Configuration

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Vocabulary Size         |              10,000 |
| Maximum Sequence Length |                 500 |
| Embedding Dimension     |                 128 |
| Recurrent Units         |                  64 |
| Dropout                 |                 0.2 |
| Batch Size              |                  64 |
| Maximum Epochs          |                  15 |
| Optimizer               |                Adam |
| Loss Function           | Binary Crossentropy |
| Output Activation       |             Sigmoid |

The only architectural component changed between the models is the recurrent layer.

### RNN

```text
Embedding → SimpleRNN(64) → Dropout → Dense(1, sigmoid)
```

### LSTM

```text
Embedding → LSTM(64) → Dropout → Dense(1, sigmoid)
```

### GRU

```text
Embedding → GRU(64) → Dropout → Dense(1, sigmoid)
```

---

## 🔧 Data Preprocessing

The IMDB reviews are already represented as integer token sequences.

The preprocessing pipeline consists of:

1. Loading the IMDB dataset.
2. Limiting the vocabulary to the first 10,000 words.
3. Mapping out-of-vocabulary token IDs to `<UNK>`.
4. Analyzing review-length distribution.
5. Selecting a maximum sequence length of `500`.
6. Padding/truncating sequences to a fixed length.
7. Performing a stratified train-validation split.
8. Keeping the official test set untouched.

### Sequence Length

The training review lengths were analyzed before selecting the maximum sequence length.

The resulting configuration was:

```python
MAX_SEQUENCE_LENGTH = 500
```

Sequences longer than 500 tokens are truncated, while shorter sequences are padded with zeros.

---

## 🏋️ Training

All models are trained using the same training configuration for a fair comparison.

Early stopping is applied using validation loss:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

### Why EarlyStopping?

Training stops when the validation loss does not improve for three consecutive epochs.

The best validation-loss weights are restored after training, helping prevent the final model from being affected by later overfitting.

---

## 📊 Evaluation Metrics

The models are evaluated on the untouched IMDB test set using:

### Accuracy

Measures the proportion of correctly classified reviews.

### Precision

Measures how many reviews predicted as positive were actually positive.

### Recall

Measures how many actual positive reviews were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between positive and negative classes across classification thresholds.

### Confusion Matrix

Shows:

```text
                 Predicted
                 Negative  Positive
Actual Negative     TN        FP
Actual Positive     FN        TP
```

---

## 📈 Results

The final test-set results obtained from the experiment are:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ----- | -------: | --------: | -----: | -------: | ------: | 
| RNN   |   0.7980 |    0.7939 | 0.8051 |   0.7995 |  0.8718 |      
| LSTM  |   0.8659 |    0.9011 | 0.8220 |   0.8597 |  0.9403 |      
| GRU   |   0.8723 |    0.9037 | 0.8334 |   0.8671 |  0.9423 |      

> Training time is included as an additional computational comparison. It is not an evaluation metric specified in the problem statement.

---

## 📌 Best Validation-Loss Epoch

EarlyStopping monitors `val_loss`, so the restored model weights correspond to the epoch with the lowest validation loss.

| Model | Best Validation-Loss Epoch |
| ----- | -------------------------: |
| RNN   |                          2 |
| LSTM  |                          2 |
| GRU   |                          3 |

This is important because the epoch with the highest validation accuracy does not necessarily correspond to the epoch with the lowest validation loss.

---

## 🔍 Model Comparison

### RNN

The Simple RNN provides a baseline for sequence classification.

It has the smallest parameter count among the three models and the shortest recorded training time in this experiment. However, its test performance was lower than that of the LSTM and GRU.

### LSTM

The LSTM introduces memory cells and gating mechanisms designed to handle longer-term dependencies in sequential data.

It achieved substantially higher classification performance than the Simple RNN, while requiring more parameters and training time.

### GRU

The GRU uses a simpler gated architecture than LSTM while still providing mechanisms for handling dependencies across sequences.

In this experiment, the GRU achieved:

* Accuracy: **87.23%**
* F1-Score: **86.71%**
* ROC-AUC: **94.23%**

Its recorded training time was lower than the LSTM while maintaining comparable classification performance.

---

## 📉 Visualizations

The notebook includes the following visualizations:

### 1. Review Length Distribution

Used to understand the distribution of review lengths and justify the selected sequence length.

### 2. Training vs Validation Accuracy

Learning curves for:

* RNN
* LSTM
* GRU

### 3. Training vs Validation Loss

Used to analyze convergence and potential overfitting.

### 4. Confusion Matrices

Separate confusion matrices are provided for all three models.

### 5. Classification Metrics Comparison

A grouped bar chart compares:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### 6. Parameter Count Comparison

Compares the number of trainable parameters across the three architectures.

---

## 🧮 Parameter Analysis

The models share the same embedding layer:

```text
Embedding parameters = 10,000 × 128
                     = 1,280,000
```

The remaining parameter differences come primarily from the recurrent layers.

| Model | Total Parameters |
| ----- | ---------------: |
| RNN   |        1,292,417 |
| LSTM  |        1,329,473 |
| GRU   |        1,317,313 |

This provides an additional comparison of model complexity.

---

## 🗂️ Project Structure

A recommended repository structure is:

```text
Assignment 7/
│
├── data/
│   └── imdb.npz
│
├── notebooks/
│   └── Assignment7.ipynb
│
└── README.md
```

> The IMDB dataset should generally **not be committed to GitHub**. The notebook can download/load the dataset through TensorFlow Keras when required.

---

## ⚙️ Requirements

Recommended environment:

* Python 3.12+
* TensorFlow 2.21.0
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

Example installation:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 🚀 Running the Notebook

Clone the repository:

```bash
git clone https://github.com/Nitish-0710/deep-learning-assignments.git
```

Navigate to the assignment directory:

```bash
cd Assignment 7
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Assignment7.ipynb
```

Run the notebook cells sequentially.

---

## 🔬 Experimental Workflow

The complete workflow implemented in the notebook is:

```text
Load IMDB Dataset
        ↓
Analyze Dataset
        ↓
Decode Sample Review
        ↓
Analyze Review Lengths
        ↓
Select Sequence Length
        ↓
Limit Vocabulary
        ↓
Train / Validation Split
        ↓
Pad / Truncate Sequences
        ↓
Build RNN
        ↓
Build LSTM
        ↓
Build GRU
        ↓
Train Models
        ↓
EarlyStopping
        ↓
Evaluate on Test Set
        ↓
Calculate Metrics
        ↓
Plot Learning Curves
        ↓
Plot Confusion Matrices
        ↓
Compare Models
        ↓
Analyze Parameters & Computation
```

---

## 📚 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📝 Conclusion

This assignment implements and compares Simple RNN, LSTM, and GRU architectures for binary sentiment classification on the IMDB Movie Reviews dataset.

Under the common experimental configuration, the Simple RNN produced lower test-set classification metrics, while the LSTM and GRU produced substantially higher performance. The experiment also demonstrates the differences in model parameter counts and recorded training time.

The results show how different recurrent architectures can affect sequence-classification performance while keeping the dataset, preprocessing pipeline, and major training settings consistent.

---

## 👨‍💻 Author

**Nitish Sahu**

B.Tech — Computer Science & Engineering (Artificial Intelligence)
Vishwakarma Institute of Technology, Pune