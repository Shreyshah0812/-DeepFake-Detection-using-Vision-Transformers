
# DeepFake Detection using Vision Transformers

A video-based DeepFake detection system built using transfer learning with Vision Transformers (ViT-B/16). The model classifies videos as REAL or FAKE by extracting and aggregating temporal features from multiple frames.

---

## 🚀 Project Overview

This project focuses on detecting manipulated videos using deep learning and transfer learning techniques.

The objectives:

- Extract facial frames from videos
- Learn spatial features using pretrained Vision Transformers
- Aggregate temporal information across frames
- Classify videos as REAL or FAKE
- Evaluate model performance on unseen data

---

## 📂 Dataset

- Real and Fake video samples
- Up to 200 REAL and 200 FAKE videos used
- 16 frames sampled per video
- 70% Train | 15% Validation | 15% Test split

---

## 🧠 Model Architecture

### 🔹 Feature Extraction
- Pretrained Vision Transformer (ViT-B/16)
- Frozen backbone (ImageNet weights)
- Frame-level embedding extraction (768-dim features)

### 🔹 Temporal Aggregation
- Frame averaging baseline
- Optional Transformer encoder head for sequence modeling

### 🔹 Classification
- Fully connected layer
- Sigmoid activation for binary classification
- BCEWithLogitsLoss

---

## ⚙️ Training Configuration

- Optimizer: AdamW
- Weight Decay Regularization
- 10 Training Epochs
- Batch Size: 1
- Sequence Length: 16 Frames
- Validation-based checkpointing

---

## 📊 Model Evaluation

- Precision
- Recall
- F1-Score
- Confusion Matrix
- Held-out Test Set Evaluation

---

## 🖥 Inference

The system includes:

- Single-video prediction function
- Confidence scoring
- Real-time frame visualization support

---

## 🛠 Tech Stack

Python | PyTorch | Torchvision | NumPy | OpenCV | Scikit-learn

---

## 📈 Key Outcomes

- Built an end-to-end video classification pipeline
- Implemented transfer learning for efficient training
- Designed modular Dataset & DataLoader architecture
- Enabled scalable DeepFake prediction workflow
