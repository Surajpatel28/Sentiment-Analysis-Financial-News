import streamlit as st
import joblib
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import re
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Monkeypatch torch.load to force CPU mapping globally ---
# Critical Fix: The BERT joblib file contains nested torch storages that trigger recursive 
# torch.load calls without map_location, causing CUDA errors on CPU.
original_torch_load = torch.load
def safe_cpu_load(*args, **kwargs):
    # Determine if we are on a CPU-only machine? Yes, strictly for this app.
    # We force 'map_location' to cpu if not present.
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return original_torch_load(*args, **kwargs)
torch.load = safe_cpu_load
# ------------------------------------------------------------

MODEL_FILES = {
    "bert_financial_model.joblib": "152Ds1Jr_kYUQPcfzeZkxOpabp6vw3SPI", 
    # "lstm_sentiment_model.joblib": "ANOTHER_ID_IF_NEEDED",
}

def download_model_if_missing(filename, file_id):
    if not os.path.exists(filename):
        if file_id == "YOUR_GDRIVE_FILE_ID_HERE":
            st.error(f"Model file '{filename}' is missing and no download ID is provided. Please update 'MODEL_FILES' in app.py with your Google Drive File ID.")
            return False
            
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        # Use print instead of st.write to avoid cached UI replay
        print(f"Downloading {filename} from Google Drive...")
        try:
             gdown.download(url, filename, quiet=False)
             print(f"Downloaded {filename}!")
        except Exception as e:
             st.error(f"Failed to download {filename}: {e}")
             return False
    return True

# --- Model Definitions ---

class MyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=32, output_dim=3):
        super(MyLSTM, self).__init__()

        # Embedding layer
        # padding_idx = 0 ensures that the model ignores the 0s we added for the mask
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,             # Kept to 1 for small data
            batch_first=True,
            bidirectional=True
        )

        # 3. Output Head
        # input is hidden_dim * 2 (Forward + Backward state)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text, text_lengths):
        # text shape = [batch_size, seq_len]

        # embedding shape = [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text)

        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # lstm output
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # we will use the final hidden layer of the last layer
        # concatenate the final forward and backwar hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        hidden_cat = self.dropout(hidden_cat)

        return self.fc(hidden_cat)

# --- Helper Functions ---

def preprocess_text_lstm(text, vocab, max_len=64):
    """
    Tokenizes text, maps to IDs using vocab, and pads/truncates to max_len.
    Returns tensor and actual length (clamped to max_len).
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    # Map to IDs
    token_ids = [vocab.get(token, 1) for token in tokens]
    
    actual_length = len(token_ids)
    
    # Pad or Truncate
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
        actual_length = max_len
        
    return torch.tensor([token_ids], dtype=torch.long), torch.tensor([actual_length], dtype=torch.long)

@st.cache_resource
def load_lr_model():
    try:
        data = joblib.load('sentiment_analysis_v1.joblib')
        # Check if it's the expected dict structure
        if isinstance(data, dict) and 'model' in data and 'vectorizer' in data:
            return data
        else:
             # Fallback if it was just the model
             return {'model': data, 'vectorizer': None}
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

@st.cache_resource
def load_lstm_resources():
    try:
        vocab = joblib.load('lstm_vocab.joblib')
        try:
            model_weights = joblib.load('lstm_sentiment_model.joblib')
        except:
             model_weights = torch.load('lstm_sentiment_model.joblib', map_location=torch.device('cpu'))

        # Instantiate Model Here to Return Ready-to-Use Object
        if isinstance(vocab, dict):
            actual_vocab_size = max(vocab.values()) + 1
        else:
            actual_vocab_size = 10000 
            
        model = MyLSTM(vocab_size=actual_vocab_size, output_dim=3)
        
        if isinstance(model_weights, dict):
             if 'state_dict' in model_weights:
                 model.load_state_dict(model_weights['state_dict'])
             else:
                 model.load_state_dict(model_weights)
        else:
            model = model_weights
        
        model.eval()
        return vocab, model
        
    except FileNotFoundError as e:
        return None, None
    except Exception as e:
        print(f"Detailed LSTM Load Error: {e}") 
        return None, None

@st.cache_resource
def load_bert_resources():
    try:
        model_path = 'bert_financial_model.joblib'
        if not download_model_if_missing(model_path, MODEL_FILES.get(model_path)):
             return None, None

        # Initialize Skeleton
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=3
        )
        
        # Load Weights
        # Use torch.load with map_location='cpu' to handle CUDA->CPU
        # strict=False allows ignoring missing keys if any (though state_dict shouldn't have extras)
        # weights_only=False needed for some pickled archives (like joblib/older torch)
        try:
             state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception:
             # Fallback just in case, though likely to fail if CUDA
             state_dict = joblib.load(model_path)

        model.load_state_dict(state_dict)
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        print(f"BERT Load Error: {e}")
        return None, None

def predict_bert(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, predicted = torch.max(probs, 1)
        
    return predicted.item(), conf.item()

def get_sentiment_label(prediction):
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if isinstance(prediction, str):
        return prediction
    return mapping.get(int(prediction), f"Unknown ({prediction})")

# --- Main App ---

def main():
    st.set_page_config(page_title="Financial News Sentiment Analysis", layout="wide")
    
    st.title("Financial News Sentiment Analysis")

    # --- Load Models at Startup ---
    with st.spinner("Loading models..."):
        lr_data = load_lr_model()
        lstm_vocab, lstm_model = load_lstm_resources()
        bert_tokenizer, bert_model = load_bert_resources()

    if lr_data is None:
        st.error("Failed to load Logistic Regression Model. Please check files.")
    if lstm_model is None:
        st.error("Failed to load LSTM Model. Please check files.")
    if bert_model is None:
        st.error("Failed to load BERT Model. Please check files.")
    
    
    # Main Content
    st.subheader("Enter Financial News")
    user_input = st.text_area("Paste news article or headline here...", height=200)
    
    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        st.markdown("### Analysis Results")
        col1, col2, col3 = st.columns(3)

        # --- Logistic Regression ---
        with col1:
            st.markdown("#### Logistic Regression")
            if lr_data and 'model' in lr_data:
                try:
                    model = lr_data['model']
                    vectorizer = lr_data.get('vectorizer')
                    
                    if vectorizer:
                        input_data = vectorizer.transform([user_input])
                    else:
                        input_data = [user_input] 
                        
                    prediction = model.predict(input_data)[0]
                    
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_data)
                        confidence = np.max(proba)
                    
                    sentiment = get_sentiment_label(prediction)
                    
                    if isinstance(sentiment, str) and sentiment.lower() != "unknown":
                        color = "grey"
                        if sentiment == "Positive": color = "green"
                        elif sentiment == "Negative": color = "red"
                        st.markdown(f'<div style="text-align:center; color:{color}; font-size:24px; font-weight:bold;">{sentiment}</div>', unsafe_allow_html=True)
                    else:
                        st.write(sentiment)

                    if confidence:
                         st.progress(confidence, text=f"Conf: {confidence:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                 st.error("Model unavailable.")

        # --- LSTM ---
        with col2:
            st.markdown("#### LSTM (Custom)")
            if lstm_vocab and lstm_model:
                try:
                    input_tensor, seq_length = preprocess_text_lstm(user_input, lstm_vocab)
                    with torch.no_grad():
                        outputs = lstm_model(input_tensor, seq_length)
                        probs = torch.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                        
                    prediction_idx = predicted.item()
                    confidence_score = conf.item()
                    sentiment = get_sentiment_label(prediction_idx)
                    
                    if isinstance(sentiment, str) and sentiment.lower() != "unknown":
                        color = "grey"
                        if sentiment == "Positive": color = "green"
                        elif sentiment == "Negative": color = "red"
                        st.markdown(f'<div style="text-align:center; color:{color}; font-size:24px; font-weight:bold;">{sentiment}</div>', unsafe_allow_html=True)
                    else:
                         st.write(sentiment)
                    
                    st.progress(confidence_score, text=f"Conf: {confidence_score:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model unavailable.")

        # --- BERT ---
        with col3:
            st.markdown("#### BERT (Fine-Tuned)")
            if bert_tokenizer and bert_model:
                try:
                    prediction_idx, confidence_score = predict_bert(user_input, bert_model, bert_tokenizer)
                    sentiment = get_sentiment_label(prediction_idx)
                    
                    if isinstance(sentiment, str) and sentiment.lower() != "unknown":
                        color = "grey"
                        if sentiment == "Positive": color = "green"
                        elif sentiment == "Negative": color = "red"
                        st.markdown(f'<div style="text-align:center; color:{color}; font-size:24px; font-weight:bold;">{sentiment}</div>', unsafe_allow_html=True)
                    else:
                         st.write(sentiment)
                            
                    st.progress(confidence_score, text=f"Conf: {confidence_score:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model unavailable.")

if __name__ == "__main__":
    main()
