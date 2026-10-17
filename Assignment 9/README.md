# Assignment 9 — Image Classification Using Vision Transformer and CNN

Implement image classification using a pre-trained **Vision Transformer (ViT-B/16)** and compare its performance with a **ResNet18 CNN** using PyTorch and TorchVision.

## Objective

The objective is to classify natural scene images into six categories using transfer learning and compare the classification performance of both models.

## Models

| Model | Architecture | Pre-trained Weights |
|---|---|---|
| ViT-B/16 | Vision Transformer | ImageNet |
| ResNet18 | Convolutional Neural Network | ImageNet |

Both models use a frozen pre-trained backbone and a newly trained classification head.

## Dataset

**Intel Natural Scenes Classification Dataset**

- 6 classes: Buildings, Forest, Glacier, Mountain, Sea, and Street
- 14,034 training images
- 3,000 test images
- Training data split into 80% training and 20% validation
- Images resized to `224 × 224`
- ImageNet normalization applied

Dataset source: [Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | `224 × 224` |
| Batch size | `16` |
| Epochs | `5` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Loss function | Cross-Entropy Loss |
| Weight decay | `0.0001` |

## Repository Structure

```text
Assignment 9/
├── Assignment9.ipynb
├── README.md
└── outputs/
    └── results/
        ├── model_comparison.csv
        ├── vit_b16_test_results.json
        ├── resnet18_test_results.json
        ├── normalized_confusion_matrices.png
        ├── test_performance_comparison.png
        └── training_validation_curves.png
```

The dataset and trained model checkpoints are excluded from GitHub.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `Assignment9.ipynb` in Jupyter Notebook and execute the cells sequentially.

The notebook downloads and prepares the dataset, trains both models, evaluates their performance, and generates the result files.

A CUDA-compatible GPU is recommended for faster training.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The results are saved in CSV and JSON files, along with visualizations for comparing both models.

## Conclusion

The assignment demonstrates transfer learning using a pre-trained Vision Transformer and a CNN for image classification. Their performance is compared on the Intel Natural Scenes Classification dataset using standard classification metrics.

## Author

**Nitish Charanlal Sahu**

B.Tech CSE (Artificial Intelligence)

Vishwakarma Institute of Technology, Pune
