import joblib
import numpy as np
from pathlib import Path
import gradio as gr
import json
import torch, torch.nn as nn

# ---- Load artifacts ----
MODELS = Path("models")
tfidf = joblib.load(MODELS / "tfidf.joblib")
clf   = joblib.load(MODELS / "logreg.joblib")

# ---- Load test metrics (from JSON) ----
try:
    metrics = json.load(open("experiments/baseline_metrics.json"))
    test = metrics.get("test_metrics", {})
    summary = f"**Test Results:** Acc {test.get('accuracy',0):.3f} • F1 {test.get('f1',0):.3f} • ROC-AUC {test.get('roc_auc',0):.3f}"
except Exception:
    summary = ""

# ---- Prepare explainability ----
feature_names = np.array(tfidf.get_feature_names_out())
coef = clf.coef_.ravel()

def predict_and_explain(text: str, top_k: int = 8):
    text = (text or "").strip()
    if not text:
        return "—", 0.0, "—", "—"

    X = tfidf.transform([text])
    prob_pos = float(clf.predict_proba(X)[0, 1])
    label = "Positive ✅" if prob_pos >= 0.5 else "Negative ❌"

    # Feature contributions (linear inspection)
    x = X.toarray()[0]
    contrib = x * coef
    nz_idx = np.flatnonzero(x)
    if nz_idx.size == 0:
        return label, prob_pos, "—", "—"

    nz_contrib = contrib[nz_idx]
    nz_feat = feature_names[nz_idx]

    pos_order = np.argsort(nz_contrib)[-top_k:][::-1]
    neg_order = np.argsort(nz_contrib)[:top_k]

    def fmt(idx_list, sign):
        parts = []
        for i in idx_list:
            w = nz_contrib[i]
            if (sign == "pos" and w > 0) or (sign == "neg" and w < 0):
                parts.append(f"{nz_feat[i]}  ({w:+.3f})")
        return "\n".join(parts) if parts else "—"

    return label, round(prob_pos, 4), fmt(pos_order, "pos"), fmt(neg_order, "neg")











# --- LSTM/GRU: imports ---
import json, torch, torch.nn as nn
from torchtext.data.utils import get_tokenizer

# --- class must match training ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_layers=1,
                 bidirectional=True, cell="lstm", pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        rnn_cls = nn.LSTM if cell.lower()=="lstm" else nn.GRU
        self.rnn = rnn_cls(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        emb = self.dropout(self.emb(x))
        out, _ = self.rnn(emb)
        mask = (x != 0).unsqueeze(-1)              # PAD=0
        out = out * mask
        pooled = (out.sum(1) / mask.sum(1).clamp(min=1))
        return self.fc(self.dropout(pooled))

# ---- load vocab + weights (graceful if missing) ----
MODELS = Path("models")  # you already have this above for baseline
VOCAB_PATH = MODELS / "lstm_vocab.json"
WEIGHTS_PATH = MODELS / "lstm_imdb.pt"
_tokenizer = get_tokenizer("basic_english")

def _try_load_lstm():
    if not VOCAB_PATH.exists() or not WEIGHTS_PATH.exists():
        return None, None, None, "Model files not found. Train & save LSTM (see 03_lstm_gru.ipynb)."
    itos = json.load(open(VOCAB_PATH))["itos"]
    stoi = {w:i for i,w in enumerate(itos)}
    pad_idx = itos.index("<pad>") if "<pad>" in itos else 0
    model = RNNClassifier(vocab_size=len(itos), pad_idx=pad_idx)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model, stoi, pad_idx, None

_LSTM_MODEL, _STOI, _PAD_IDX, _LSTM_ERR = _try_load_lstm()

def _encode_lstm(text, max_len=256):
    ids = [_STOI.get(tok, _STOI.get("<unk>", 1)) for tok in _tokenizer(text)]
    ids = ids[:max_len] or [_PAD_IDX]
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)       # [1, T]
    if x.size(1) < max_len:
        pad = torch.full((1, max_len - x.size(1)), _PAD_IDX, dtype=torch.long)
        x = torch.cat([x, pad], dim=1)
    return x

def lstm_predict(text: str):
    try:
        text = (text or "").strip()
        if not text:
            return "—", 0.0
        if _LSTM_ERR:
            return _LSTM_ERR, 0.0
        x = _encode_lstm(text)
        with torch.no_grad():
            logits = _LSTM_MODEL(x)
            prob_pos = torch.softmax(logits, dim=-1)[0, 1].item()  # <-- .item(), not .numpy()
        label = "Positive ✅" if prob_pos >= 0.5 else "Negative ❌"
        return label, round(prob_pos, 4)
    except Exception as e:
        print("LSTM error:", repr(e))  
        return f"Error: {e.__class__.__name__}", 0.0












from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path

DISTIL_DIR = Path("models/distilbert-imdb-full")

try:
    distil_tok = AutoTokenizer.from_pretrained(DISTIL_DIR)
    distil_model = AutoModelForSequenceClassification.from_pretrained(DISTIL_DIR)
    # Use HF pipeline instead of manual TextClassificationPipeline
    distil_pipe = pipeline("text-classification",
                           model=distil_model,
                           tokenizer=distil_tok,
                           return_all_scores=True)
    DISTIL_READY = True
except Exception as e:
    print("⚠️ DistilBERT load error:", repr(e))
    DISTIL_READY = False

def distil_predict(text: str):
    text = (text or "").strip()
    if not text:
        return "—", 0.0
    if not DISTIL_READY:
        return "Model not loaded. Train & save DistilBERT.", 0.0

    # Run pipeline → returns list of scores
    out = distil_pipe(text, truncation=True, max_length=256)[0]
    scores = {d["label"]: d["score"] for d in out}
    prob_pos = float(scores.get("LABEL_1", 0.0))
    label = "Positive ✅" if prob_pos >= 0.5 else "Negative ❌"
    return label, round(prob_pos, 4)












# ---- Gradio UI ----
with gr.Blocks(title="IMDB Sentiment — Baseline & LSTM/GRU") as demo:
    gr.Markdown("# 🎬 IMDB Sentiment Analysis")
    if summary:
        gr.Markdown(summary)

    with gr.Tab("Baseline (TF-IDF + Logistic Regression)"):
        with gr.Row():
            with gr.Column(scale=2):
                txt = gr.Textbox(lines=8, label="Paste a movie review")
                topk = gr.Slider(3, 15, value=8, step=1, label="Top tokens to show")
                btn = gr.Button("Predict")
            with gr.Column(scale=1):
                out_label = gr.Textbox(label="Prediction")
                out_prob  = gr.Number(label="Confidence (P(positive))")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔼 Positive-leaning tokens")
                out_pos = gr.Textbox(lines=12, label="Top positive contributors")
            with gr.Column():
                gr.Markdown("### 🔽 Negative-leaning tokens")
                out_neg = gr.Textbox(lines=12, label="Top negative contributors")
        btn.click(fn=predict_and_explain, inputs=[txt, topk],
                  outputs=[out_label, out_prob, out_pos, out_neg])

    with gr.Tab("LSTM/GRU"):
        l_txt = gr.Textbox(lines=8, label="Paste a movie review")
        l_btn = gr.Button("Predict")
        l_label = gr.Textbox(label="Prediction")
        l_prob  = gr.Number(label="Confidence (P(positive))")
        l_btn.click(fn=lstm_predict, inputs=[l_txt], outputs=[l_label, l_prob])


    with gr.Tab("DistilBERT (fine-tuned)"):
        d_txt = gr.Textbox(lines=8, label="Paste a movie review")
        d_btn = gr.Button("Predict")
        d_label = gr.Textbox(label="Prediction")
        d_prob  = gr.Number(label="Confidence (P(positive))")
        d_btn.click(fn=distil_predict, inputs=[d_txt], outputs=[d_label, d_prob])



if __name__ == "__main__":
    demo.launch()



