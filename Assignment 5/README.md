# Assignment 5 — Transfer Learning using Pre-trained CNN Models

Implement transfer learning using **AlexNet, VGG16, ResNet50, and EfficientNetB0** on the **Oxford Flowers 102** dataset using PyTorch and TorchVision.

## Objective

The objective is to reuse visual features learned from ImageNet and adapt the final classification layer to the 102 classes of Oxford Flowers 102.

### Transfer-learning setup

- ImageNet-pretrained weights
- Input size: `224 × 224`
- Batch size: `32`
- Pretrained feature extractor: **frozen**
- New classifier: **trainable**
- Output classes: **102**
- Optimizer: Adam
- Learning rate: `0.001`
- Epochs: `5`
- Loss: Cross-Entropy Loss

## Models

| Model | Original head replaced |
|---|---|
| AlexNet | `classifier[6]` |
| VGG16 | `classifier[6]` |
| ResNet50 | `fc` |
| EfficientNetB0 | `classifier[1]` |

The final classifier is replaced programmatically, so the same experiment pipeline can be reused for all four architectures.

## Dataset

**Oxford Flowers 102**

- 102 classes
- 1,020 training images
- 1,020 validation images
- 6,149 test images
- Images resized to `224 × 224`

Training uses random horizontal flipping and random rotation. Validation and test data use resizing and ImageNet normalization without augmentation.

## Repository Structure

```text
Assignment-5-Transfer-Learning/
│
├── Assignment5.ipynb
├── README.md
│
└── Images/
    │
    ├── AlexNet.png
    ├── EfficientB0.png
    ├── ResNet50.png
    ├── VGG16.png        
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Assignment-5-Transfer-Learning
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook Assignment5.ipynb
```

The Oxford Flowers 102 dataset is downloaded automatically by TorchVision.

## Dataset Location

The notebook uses a portable default:

```python
DATASET_DIR = Path(
    os.getenv("FLOWERS102_DATASET_DIR", "./data/flowers102")
)
```

Therefore, **no machine-specific path is committed to GitHub**.

If a custom dataset location is required, set the environment variable:

Windows PowerShell:

```powershell
$env:FLOWERS102_DATASET_DIR="D:\datasets\flowers102"
```

Linux/macOS:

```bash
export FLOWERS102_DATASET_DIR="/home/user/datasets/flowers102"
```

You can also change `NUM_WORKERS` through:

```text
FLOWERS102_NUM_WORKERS
```

For example, setting it to `0` is useful on systems where multiprocessing in Jupyter causes issues.

## Code Organization

The original version contained four repeated blocks for:

1. Model loading and classifier replacement
2. Training configuration
3. Training
4. Training/validation plots
5. Test evaluation

The GitHub-ready version removes this duplication using reusable functions:

- `count_parameters()`
- `train_model()`
- `evaluate_model()`
- `plot_history()`
- `summarize_history()`
- `build_transfer_model()`
- `run_experiment()`

All four models are then executed with:

```python
for model_name in MODEL_CONFIGS:
    results[model_name] = run_experiment(
        model_name=model_name,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
```

This makes the implementation shorter, easier to maintain, and ensures that every model follows the same training and evaluation procedure.

## Recorded Results

From the recorded experiment:

| Rank | Model | Test Accuracy |
|---:|---|---:|
| 1 | ResNet50 | **73.86%** |
| 2 | EfficientNetB0 | **70.15%** |
| 3 | AlexNet | **65.65%** |
| 4 | VGG16 | **64.14%** |

The notebook calculates the results again when executed. Exact values can vary slightly across environments.

## Architecture Diagrams

Architecture diagrams can be placed in:

```text
assets/architecture/
```

Recommended filenames:

```text
Alexnet.png
VGG16.png
ResNet50.png
EfficientNetB0.png
```

## Conclusion

The experiment demonstrates transfer learning with four popular CNN architectures. By freezing the pretrained feature extractor and training only a new 102-class classifier, the models can be adapted to Oxford Flowers 102 without training the complete networks from scratch.

In the recorded run, **ResNet50 achieved the highest test accuracy of 73.86%**.
