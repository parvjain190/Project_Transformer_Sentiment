# Modeling Temporal Evolution of Sentiment in IMDb Reviews
### Using Transformer-Based Time Series Architectures

## 📌 Overview
This project implements a **Transformer encoder** with **positional encoding** using TensorFlow and Keras to model how sentiment evolves across the temporal axis of each IMDb review[cite: 1]. Key upgrades over a plain LSTM include capturing long-range temporal dependencies and treating token positions as a time series[cite: 1]. 

---

## ✨ Key Features
* **Positional Encoding:** Treats token positions as a time series by injecting sinusoidal positional encodings so the model understands the temporal order of tokens[cite: 1].
* **Multi-Head Self-Attention:** Models temporal dependencies between any two positions to capture long-range temporal dependencies[cite: 1].
* **Attention Weight Extraction:** Includes a specialized sub-model (`TransformerBlockWithAttn`) to extract self-attention weights for interpretability[cite: 1].
* **Custom Review Prediction:** A pipeline to tokenize, pad, and predict sentiment probabilities on custom text[cite: 1].

---

## 🛠️ Technology Stack
* **Framework:** TensorFlow 2.16.1 / Keras[cite: 1]
* **Data Processing:** NumPy, Keras `sequence` preprocessing, IMDb dataset[cite: 1]
* **Evaluation:** Scikit-learn (`classification_report`, `confusion_matrix`, `accuracy_score`)[cite: 1]
* **Visualization:** Matplotlib, Seaborn[cite: 1]

---

## 🏗️ Model Architecture
The model leverages the Keras Functional API with the following structure[cite: 1]:

| Layer Type | Description | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Token sequence | `(None, 200)` | 0[cite: 1] |
| **TokenAndPositionEmbedding** | Token + Positional Encoding | `(None, 200, 64)` | 652,800[cite: 1] |
| **TransformerBlock 0** | Multi-Head Attention + Add/Norm + Feed-Forward | `(None, 200, 64)` | 33,472[cite: 1] |
| **TransformerBlock 1** | Multi-Head Attention + Add/Norm + Feed-Forward | `(None, 200, 64)` | 33,472[cite: 1] |
| **GlobalAveragePooling1D** | Aggregate temporal dimension | `(None, 64)` | 0[cite: 1] |
| **Dense (ReLU) & Dropout** | Fully connected hidden layer (Dropout 0.1) | `(None, 64)` | 4,160[cite: 1] |
| **Dense (Sigmoid)** | Sentiment probability output | `(None, 1)` | 65[cite: 1] |

---

## 📊 Configuration & Hyperparameters
* **Vocab Size:** 10,000[cite: 1]
* **Max Length:** 200[cite: 1]
* **Embedding Dimension:** 64[cite: 1]
* **Attention Heads:** 4[cite: 1]
* **Feed-Forward Dimension:** 128[cite: 1]
* **Transformer Blocks:** 2[cite: 1]
* **Dropout Rate:** 0.1[cite: 1]
* **Batch Size:** 32[cite: 1]
* **Epochs:** 5[cite: 1]

---

## 🚀 Results & Performance
Trained on 25,000 samples and evaluated on 25,000 test samples[cite: 1]. 
* **Final Accuracy:** 85.08%[cite: 1]
* **Precision / Recall / F1-Score:** 0.85 across both Negative and Positive classes[cite: 1]
* **Training Time:** ~230-260 seconds per epoch[cite: 1]

---

## 💡 Custom Prediction Examples
The model can predict sentiment on custom strings[cite: 1]:

> **Review:** "This movie was fantastic and the acting was great"[cite: 1]
> **Prediction:** POSITIVE (confidence: 0.9979)[cite: 1]

> **Review:** "Boring and predictable with terrible dialogue"[cite: 1]
> **Prediction:** NEGATIVE (confidence: 0.0001)[cite: 1]
