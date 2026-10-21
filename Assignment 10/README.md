# Assignment 10: Audio Classification Using Deep Learning with STT and TTS

## Overview

This assignment implements a deep learning application for audio classification using a Convolutional Neural Network (CNN). The application classifies spoken commands from audio recordings and integrates Speech-to-Text (STT) using OpenAI Whisper and Text-to-Speech (TTS) using `pyttsx3`.

The project uses the **Google Speech Commands v0.02** dataset and demonstrates audio preprocessing, spectrogram-based feature extraction, CNN training, model evaluation, speech recognition, and speech synthesis.

## Objectives

- Explore and preprocess an audio classification dataset.
- Convert audio recordings into log-Mel spectrograms.
- Design and train a CNN for spoken-command classification.
- Evaluate model performance using appropriate classification metrics.
- Integrate a pretrained Whisper model for Speech-to-Text.
- Generate spoken responses using Text-to-Speech.
- Analyze the results and limitations of the implemented system.

## Dataset

**Dataset:** Google Speech Commands v0.02

The experiment uses ten spoken-command categories:

`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, and `go`.

The dataset contains **38,546 WAV recordings** across these ten categories.

### Dataset split

| Split          | Number of samples |
| -------------- | ----------------: |
| Training       |            20,000 |
| Validation     |             5,782 |
| Testing        |             5,782 |
| **Total used** |        **31,564** |

The training set contains 2,000 samples per command. A stratified fallback split was used because the official validation and testing lists were not found in the extracted dataset. File-level overlap checks confirmed that no audio file paths were shared between the three splits.

The training subset is capped at 2,000 recordings per class to keep computation manageable.

## Technologies Used

- **Programming language:** Python
- **Deep learning framework:** PyTorch
- **Audio processing:** Librosa, SoundFile
- **Numerical computing:** NumPy
- **Data handling and visualization:** Pandas, Matplotlib
- **Model evaluation:** Scikit-learn
- **Speech-to-Text:** OpenAI Whisper (`base.en`)
- **Text-to-Speech:** `pyttsx3`
- **Execution environment:** Jupyter Notebook with CUDA-enabled GPU support

## Methodology

### 1. Audio preprocessing

Audio recordings are loaded as mono signals, resampled to 16 kHz, and padded or truncated to one second.

### 2. Feature extraction

Each audio recording is converted into a **64-bin log-Mel spectrogram**. This time-frequency representation helps the CNN learn patterns associated with different spoken commands.

### 3. CNN architecture

The CNN consists of three convolutional blocks containing convolution, batch normalization, ReLU activation, and max-pooling layers. Adaptive average pooling produces a fixed-size feature representation, which is passed to fully connected layers with dropout to classify the input into one of ten command categories.

### 4. Model training

The model is trained using:

- **Optimizer:** Adam
- **Loss function:** Cross-Entropy Loss
- **Learning rate:** 0.001
- **Batch size:** 64
- **Maximum epochs:** 100
- **Early stopping patience:** 5 epochs
- **Hardware acceleration:** CUDA when available

The model is evaluated on the validation set after each epoch. The weights associated with the lowest validation loss are restored and saved for final testing.

### 5. Model evaluation

Performance is evaluated on the held-out test set using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Classification report
- Confusion matrix

Training and validation loss and accuracy curves are plotted to analyze learning behavior and generalization.

### 6. Speech-to-Text

The pretrained Whisper `base.en` model transcribes audio recordings into text. STT performance is evaluated on a reproducible subset of 30 test recordings using normalized Word Error Rate (WER), exact command match rate, and mean transcription latency.

### 7. Text-to-Speech

The recognized text is passed to `pyttsx3`, which synthesizes a spoken response and saves it as a WAV file.

## Results

### CNN classification performance

The CNN completed training after 30 epochs, when early stopping was triggered.

| Metric           | Result |
| ---------------- | -----: |
| Test accuracy    | 94.02% |
| Macro precision  | 94.09% |
| Macro recall     | 94.00% |
| Macro F1-score   | 94.02% |
| Test samples     |  5,782 |
| Epochs completed |     30 |

The confusion matrix and classification report provide class-wise performance details. The model achieved strong classification performance across the ten spoken commands.

### Speech-to-Text performance

| Metric                   |                      Result |
| ------------------------ | --------------------------: |
| Whisper model            |                   `base.en` |
| Evaluation samples       |                          30 |
| Normalized WER           |                      23.33% |
| Exact command match rate |                      76.67% |
| Mean STT latency         | Approximately 0.092 seconds |

These STT results represent the evaluated subset of 30 recordings and should not be interpreted as measurements across the entire test dataset.

### Text-to-Speech

The TTS component successfully generated a WAV audio response from the recognized text. The measured synthesis latency was approximately 0.136 seconds in the reported run.

## Project Structure

```text
Assignment 10/
├── Assignment10.ipynb
├── README.md
├── results/
    ├── assignment10_metrics.json
    └── speech_response.wav
```

The dataset is extracted into the user's home directory by default to avoid slow extraction on the mounted Windows filesystem. The notebook can also use an existing dataset in the assignment directory.

The `results/` directory contains the trained model checkpoint, experiment metrics, and generated audio when these artifacts have been saved.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Nitish-0710/deep-learning-assignments.git
cd deep-learning-assignments
```

Navigate to the Assignment 10 directory.

### 2. Install dependencies

Run the following in your active Python environment:

```bash
pip install torch librosa soundfile numpy pandas matplotlib scikit-learn requests tqdm openai-whisper jiwer pyttsx3 jupyter
```

Whisper may require FFmpeg. On Ubuntu/WSL, install the required system packages if they are missing:

```bash
sudo apt update
sudo apt install -y ffmpeg espeak-ng
```

Open `Assignment10.ipynb` and execute the cells sequentially.

The notebook automatically selects CUDA when a compatible GPU is available and otherwise falls back to the CPU. The first run may download the dataset and pretrained Whisper model.

## Limitations

- The classifier recognizes only ten predefined spoken commands.
- Audio is limited to one-second inputs.
- Training uses a capped subset of the available recordings.
- A stratified fallback split is used because the official split lists were unavailable.
- File-level separation does not guarantee speaker-independent evaluation.
- Whisper is evaluated on only 30 samples, so its reported metrics are preliminary.
- STT and TTS latency depends on hardware, model initialization, and the execution environment.

## Conclusion

This assignment demonstrates a complete audio-processing pipeline combining CNN-based spoken-command classification, pretrained Whisper speech recognition, and speech synthesis using `pyttsx3`.

The CNN achieved **94.02% test accuracy and a 94.02% macro F1-score** on the held-out test set. The STT component achieved a 76.67% exact command match rate on the 30-sample evaluation subset, while the TTS component successfully generated spoken output.

The experiment illustrates how deep learning and pretrained speech models can be combined to build an audio classification and speech interaction application.

## Author

**Nitish Sahu**  
Vishwakarma Institute of Technology (VIT), Pune  
B.Tech — Computer Science and Engineering (AI Specialization)

GitHub: [Nitish-0710](https://github.com/Nitish-0710)

## License and Dataset Attribution

The dataset is provided by the Google Speech Commands project. Refer to the dataset's original distribution and terms for usage and attribution details. This assignment's code is intended for educational purposes.
