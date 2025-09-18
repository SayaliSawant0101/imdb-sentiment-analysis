# 🎬 IMDB Sentiment Analysis — Sayali Sawant

**Live Demo:** [👉 Try it here](https://huggingface.co/spaces/sayalis2024/imdb-sentiment-sayali)

This project predicts whether a movie review is **Positive ✅** or **Negative ❌**, using three progressively more advanced NLP and LLM models:

1. **Baseline:** TF-IDF + Logistic Regression  
   - Simple bag-of-words features  
   - Achieved ~88% test accuracy  
   - Fast and interpretable (shows top positive/negative tokens)

2. **Deep Learning:** LSTM/GRU (PyTorch)  
   - Uses word sequences instead of bag-of-words  
   - Captures context like word order and phrasing  
   - Achieved ~85–90% test accuracy

3. **Large Language Model (LLM): DistilBERT**  
   - Fine-tuned transformer from Hugging Face  
   - Learns contextual word embeddings from pretraining  
   - Achieved ~90–91% test accuracy  
   - Handles longer text and subtle cues better than the baseline

---

## 🚀 Features in the Demo
- **Interactive Tabs:** Switch between Baseline, LSTM, and DistilBERT models.  
- **Confidence Scores:** See probability of positive sentiment.  
- **Explainability (Baseline):** Displays top words contributing to prediction.  
- **Live Input:** Paste your own movie review and test instantly.  

---

## 📊 Metrics

| Model                | Accuracy | F1 Score | ROC-AUC |
|----------------------|----------|----------|---------|
| TF-IDF + LogisticReg | 88.5%    | 0.885    | 0.95    |
| LSTM/GRU (PyTorch)   | 85–90%   | 0.90     | —       |
| DistilBERT (HF Hub)  | 90–91%   | 0.91     | —       |

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Scikit-learn, PyTorch, Transformers, Gradio  
- **Deployment:** Hugging Face Spaces (Gradio app)  
- **Training:** Custom preprocessing pipeline, HF Trainer API  

---

## 📂 Repo Structure
```
The repository is organized as follows:

imdb-sentiment-analysis/
├── app.py                # Gradio app with 3 tabs (Baseline, LSTM/GRU, DistilBERT)
├── requirements.txt      # Minimal dependencies
├── models/               # Saved models
│   ├── logreg.joblib     # TF-IDF + Logistic Regression model
│   ├── lstm_imdb.pt      # Trained LSTM/GRU model
│   └── lstm_vocab.json   # Vocabulary for LSTM
├── notebooks/            # Jupyter notebooks for training & experiments
│   ├── 01_preprocessing.ipynb
│   ├── 02_lstm_training.ipynb
│   └── 03_distilbert_finetune.ipynb
├── scripts/              # Data preparation & utility scripts
│   └── prepare_data.py
└── README.md             # Project description & docs


---
