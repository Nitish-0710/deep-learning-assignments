# Assignment 8: Pre-trained BERT for Sentiment Analysis

## Objective

Implement a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using a text classification dataset.

The assignment demonstrates how a pre-trained Transformer-based language model can be fine-tuned for a downstream binary classification task.

## Dataset

The **Stanford Sentiment Treebank (SST-2)** dataset is used for binary sentiment classification.

Each sample contains a sentence with one of two sentiment labels:

- `0` → Negative
- `1` → Positive

The dataset contains separate training, validation, and test splits.

## Model

A pre-trained **BERT Base Uncased (`bert-base-uncased`)** model is fine-tuned for sentiment classification.

The model consists of:

- Pre-trained BERT Transformer encoder
- Sequence classification head
- 2 output classes: Negative and Positive

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Methodology

The assignment follows these steps:

1. Load and explore the SST-2 dataset.
2. Analyze the class distribution.
3. Analyze sentence and BERT token lengths.
4. Load the pre-trained BERT tokenizer.
5. Tokenize the dataset using `bert-base-uncased`.
6. Prepare the dataset for PyTorch and Hugging Face `Trainer`.
7. Load the pre-trained BERT model with a binary classification head.
8. Configure the fine-tuning parameters.
9. Fine-tune BERT on the SST-2 training set.
10. Evaluate the model on the validation set.
11. Visualize training and validation performance.
12. Generate a confusion matrix.
13. Generate a classification report.
14. Perform sentiment prediction on custom sentences.
15. Analyze the final model performance.

## Reproducibility

A fixed random seed of `42` is used wherever applicable to improve reproducibility.

The computation device is detected automatically. CUDA-enabled GPU acceleration is used when available.

## BERT Configuration

| Parameter                   |               Value |
| --------------------------- | ------------------: |
| Model                       | `bert-base-uncased` |
| Number of Classes           |                   2 |
| Maximum Sequence Length     |                 128 |
| Epochs                      |                   3 |
| Batch Size per Device       |                   8 |
| Gradient Accumulation Steps |                   2 |
| Effective Batch Size        |                  16 |
| Evaluation Batch Size       |                  16 |
| Learning Rate               |              `2e-5` |
| Weight Decay                |              `0.01` |
| Warmup Steps                |                 500 |
| LR Scheduler                |              Linear |
| Mixed Precision             |                FP16 |
| Random Seed                 |                  42 |

## Evaluation Metrics

The model is evaluated using:

- **Accuracy** — proportion of correctly classified samples.
- **Precision** — proportion of predicted positive samples that are actually positive.
- **Recall** — proportion of actual positive samples correctly identified.
- **F1 Score** — harmonic mean of precision and recall.
- **Validation Loss** — classification loss calculated on the validation set.

## Results

The fine-tuned BERT model achieved the following results on the SST-2 validation set:

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **92.55%** |
| Precision       | **91.83%** |
| Recall          | **93.69%** |
| F1 Score        | **92.75%** |
| Validation Loss | **0.3848** |

### Classification Performance

The confusion matrix was:

|                     | Predicted Negative | Predicted Positive |
| ------------------- | -----------------: | -----------------: |
| **Actual Negative** |                391 |                 37 |
| **Actual Positive** |                 28 |                416 |

Class-wise performance:

| Class    | Precision | Recall | F1 Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Negative |    93.32% | 91.36% |   92.33% |     428 |
| Positive |    91.83% | 93.69% |   92.75% |     444 |

## Training Analysis

The training loss decreased consistently throughout fine-tuning.

The validation loss increased after the first epoch, indicating some degree of overfitting. However, validation accuracy and F1 score continued to improve across the three epochs.

The best recorded validation performance was obtained at epoch 3:

- Accuracy: **92.55%**
- F1 Score: **92.75%**

## Sample Predictions

The fine-tuned model was also tested on custom unseen sentences.

Example predictions included:

| Sentence                                                            | Prediction |
| ------------------------------------------------------------------- | ---------- |
| This movie was absolutely wonderful and I loved every minute of it. | Positive   |
| The film was boring, predictable, and disappointing.                | Negative   |
| An enjoyable story with excellent performances.                     | Positive   |
| I would not recommend this movie to anyone.                         | Negative   |

The model correctly classified all four demonstration sentences.

## Project Structure

```text
Assignment-8-BERT-Sentiment-Analysis/
│
├── Assignment_8_BERT_Sentiment_Analysis.ipynb
├── README.md
└── results/
    └── bert-sst2/
```

> The `results/` directory contains the checkpoints and training outputs generated during fine-tuning.

## Conclusion

In this assignment, a pre-trained BERT model (`bert-base-uncased`) was fine-tuned for binary sentiment classification using the SST-2 dataset.

The experiment included dataset exploration, tokenization, model fine-tuning, validation-based evaluation, performance visualization, confusion matrix analysis, classification reporting, and custom sentiment inference.

The model achieved **92.55% validation accuracy** and an **F1 score of 92.75%**, demonstrating the effectiveness of transfer learning with a pre-trained BERT model for sentiment classification.
