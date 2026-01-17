# 📈 Financial News Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

A comprehensive Machine Learning project to classify financial news headlines into **Positive**, **Negative**, or **Neutral** sentiment. This project benchmarks three different architectures—from simple statistical methods to state-of-the-art Transformers—and deploys them via an interactive Web UI.

## 🚀 Project Overview

Financial sentiment analysis is critical for algorithmic trading and market monitoring. This project implements and compares three distinct approaches:

1.  **Logistic Regression (Baseline):** A lightweight TF-IDF based model.
2.  **Bi-Directional LSTM:** A Deep Learning model using custom embeddings to capture sequence context.
3.  **Fine-Tuned BERT:** A state-of-the-art Transformer model (`bert-base-uncased`) fine-tuned specifically on financial data.

The final model is deployed using **Streamlit**, allowing users to paste news text and get real-time sentiment predictions with confidence scores.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Machine Learning:** Scikit-Learn
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Serialization:** Joblib

## 📂 Dataset

The models were trained on the **Financial PhraseBank** dataset (Malo et al. 2014), which contains 4,840 sentences from financial news selected from the LexisNexis database.
* **Classes:** Negative (0), Neutral (1), Positive (2)
* **Split:** Stratified Split (Train/Test)

## 🏗️ Architecture & Performance

| Model | Architecture Highlights | Use Case |
| :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF Vectorization + Logistic Regression | Ultra-fast baseline, interpretable. |
| **LSTM** | Embedding Layer → Bi-Directional LSTM (Hidden Dim 32) → Dropout → Linear | Captures sequence order and context better than LR. |
| **BERT (Fine-Tuned)** | `bert-base-uncased` with frozen encoder (Layers 0-7) and trained heads (Layers 8-11). | **State-of-the-Art accuracy.** Understands complex context (e.g., "loss narrowed"). |

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/financial-sentiment-analysis.git](https://github.com/your-username/financial-sentiment-analysis.git)
cd financial-sentiment-analysis