# Modeling Temporal Evolution of Sentiment in IMDb Reviews
### Using Transformer-Based Time Series Architectures

## 📌 Overview
This project implements a **Transformer encoder** with **positional encoding** using TensorFlow and Keras to model how sentiment evolves across the temporal axis of each IMDb review[cite: 1]. Key upgrades over a plain LSTM include capturing long-range temporal dependencies and treating token positions as a time series.

---

## ✨ Key Features
* **Positional Encoding:** Treats token positions as a time series by injecting sinusoidal positional encodings so the model understands the temporal order of tokens.
* **Multi-Head Self-Attention:** Models temporal dependencies between any two positions to capture long-range temporal dependencies.
* **Attention Weight Extraction:** Includes a specialized sub-model (`TransformerBlockWithAttn`) to extract self-attention weights for interpretability.
* **Custom Review Prediction:** A pipeline to tokenize, pad, and predict sentiment probabilities on custom text.

---

## 🛠️ Technology Stack
* **Framework:** TensorFlow 2.16.1 / Keras
* **Data Processing:** NumPy, Keras `sequence` preprocessing, IMDb dataset
* **Evaluation:** Scikit-learn (`classification_report`, `confusion_matrix`, `accuracy_score`)
* **Visualization:** Matplotlib, Seaborn

---

## 🏗️ Model Architecture
The model leverages the Keras Functional API with the following structure
| Layer Type | Description | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Token sequence | `(None, 200)` | 0 |
| **TokenAndPositionEmbedding** | Token + Positional Encoding | `(None, 200, 64)` | 652,800 |
| **TransformerBlock 0** | Multi-Head Attention + Add/Norm + Feed-Forward | `(None, 200, 64)` | 33,472 |
| **TransformerBlock 1** | Multi-Head Attention + Add/Norm + Feed-Forward | `(None, 200, 64)` | 33,472 |
| **GlobalAveragePooling1D** | Aggregate temporal dimension | `(None, 64)` | 0 |
| **Dense (ReLU) & Dropout** | Fully connected hidden layer (Dropout 0.1) | `(None, 64)` | 4,160 |
| **Dense (Sigmoid)** | Sentiment probability output | `(None, 1)` | 65 |

---

## 📊 Configuration & Hyperparameters
* **Vocab Size:** 10,000
* **Max Length:** 200
* **Embedding Dimension:** 64
* **Attention Heads:** 4
* **Feed-Forward Dimension:** 128
* **Transformer Blocks:** 2
* **Dropout Rate:** 0.1
* **Batch Size:** 32
* **Epochs:** 5
---

## 🚀 Results & Performance
Trained on 25,000 samples and evaluated on 25,000 test samples. 
* **Final Accuracy:** 85.08%
* **Precision / Recall / F1-Score:** 0.85 across both Negative and Positive classes
* **Training Time:** ~230-260 seconds per epoch

---

## 💡 Custom Prediction Examples
The model can predict sentiment on custom strings:

> **Review:** "This movie was fantastic and the acting was great"
> **Prediction:** POSITIVE (confidence: 0.9979)

> **Review:** "Boring and predictable with terrible dialogue"
> **Prediction:** NEGATIVE (confidence: 0.0001)
