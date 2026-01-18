# Financial News Sentiment Analysis

A comparative NLP project implementing **three sentiment analysis architectures** on financial news headlines: a statistical baseline, a custom deep learning model, and a Transformer-based model. The project emphasizes **data handling, class imbalance, model benchmarking, and deployment-ready artifacts**.

---

## Dataset

- **Source**: Kaggle – *Sentiment Analysis for Financial News* (Financial PhraseBank)
- **File Used**: `all-data.csv`
- **Samples**: 4,845 sentences
- **Labels**:
  - Negative → `0`
  - Neutral → `1`
  - Positive → `2`

### Class Distribution

- Neutral: **~59.4%**
- Positive: **~28.1%**
- Negative: **~12.5%**

Stratified splitting is used throughout all experiments to preserve this imbalance.

---

## Text Preprocessing

Minimal, controlled preprocessing is applied to avoid destroying financial semantics:

- Lowercasing
- URL and HTML tag removal
- Removal of special characters (letters retained)
- No stopword removal (to preserve negation)

---

## Model 1 — Logistic Regression (Baseline)

### Feature Engineering

- **TF-IDF Vectorization**
  - N-grams: `(1, 3)`
  - `min_df = 5`
  - `max_features = 10,000`
- Vectorizer fit **only on training data** to prevent leakage

### Model Configuration

- Multinomial Logistic Regression
- `class_weight='balanced'`
- Regularization strength: `C = 0.1`
- Solver: `lbfgs`
- `max_iter = 1000`

### Data Split

- Train: 80% (3,876 samples)
- Test: 20% (969 samples)
- Stratified by sentiment label

### Performance

- **Train Accuracy**: ~80.9%
- **Test Accuracy**: ~74.1%

Neutral class dominates performance; minority negative class remains hardest.

### Artifact

- Saved as: `sentiment_analysis_v1.joblib`
- Includes:
  - TF-IDF Vectorizer
  - Trained Logistic Regression model
  - Class mapping

---

## Model 2 — Bi-Directional LSTM (From Scratch)

### Vocabulary Construction

- Tokenization via regex (`\w+`)
- Words appearing fewer than **2 times removed**
- Special tokens:
  - `<PAD>` → `0`
  - `<UNK>` → `1`

### Dataset Pipeline

- Custom `torch.utils.data.Dataset`
- Dynamic sequence length handling
- Padding and truncation to `max_len = 64`
- Length-aware batching using `pack_padded_sequence`

### Architecture

- Embedding Dimension: `100`
- Bi-Directional LSTM
  - Hidden Size: `32`
  - Layers: `1`
- Dropout: `0.3`
- Fully Connected Output Head (3 classes)

### Training Setup

- Optimizer: Adam (`lr = 0.001`)
- Loss: CrossEntropyLoss
- Epochs: `5`
- Batch Size: `32`
- Evaluation Metric: **Macro F1** (handles imbalance)

### Results (Validation)

- Best Val Accuracy: **~69.9%**
- Best Macro F1: **~0.55**

LSTM captures negation and short-term context better than TF-IDF but struggles with complex financial phrasing.

### Artifacts

- `lstm_sentiment_model.joblib` (state_dict)
- `lstm_vocab.joblib`

---

## Model 3 — BERT (Transformer)

### Model

- `bert-base-uncased`
- HuggingFace Transformers
- Fine-tuned using `Trainer`

### Tokenization

- Dynamic padding via `DataCollatorWithPadding`
- Truncation enabled

### Training Configuration

- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `1`
- Weight Decay: `0.01`
- Metric: F1 score (`evaluate` library)

### Notes

- Demonstrates strong performance even with **minimal fine-tuning**
- Requires significantly less feature engineering
- Most robust to financial phrasing like *"loss narrowed"*

---

## Key Observations

- **Class imbalance dominates evaluation** — accuracy alone is misleading
- **TF-IDF + Logistic Regression** is fast, interpretable, and competitive
- **LSTM** improves contextual understanding but is data-hungry
- **BERT** generalizes best with minimal task-specific tuning

---

## License

MIT License
