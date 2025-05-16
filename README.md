---

# 🚫 Toxic Comment Classification with DistilBERT

Welcome! 👋 This project is a hands-on deep learning pipeline built in Google Colab for classifying toxic comments using the powerful **DistilBERT** language model. We fine-tune a multilingual transformer on a real-world dataset from the [Jigsaw Toxic Comment Challenge](https://www.kaggle.com/code/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert/notebook) to help automatically detect online toxicity in comments.

---

Google Colab Link - https://drive.google.com/file/d/1CqCUVrRnqad9yw31ipC-frzfAuC-6kxF/view?usp=sharing

---

## 📚 What This Project Covers

This notebook walks you through every stage of a modern NLP pipeline:

1. **Environment Setup** – Mount Google Drive, clone the repo, set paths, and check GPU availability.
2. **Data Handling** – Load and process a subset of the Jigsaw dataset for training, validation, and testing.
3. **Preprocessing** – Tokenize and encode comments using HuggingFace's `distilbert-base-multilingual-cased` tokenizer.
4. **Model Building** – Use a distilled BERT transformer with a binary classification head.
5. **Training Loop** – Train with class imbalance handling using weighted BCE loss and evaluate with accuracy & AUC.
6. **Evaluation** – Test on multilingual and English-only subsets, and visualize performance with confusion matrices.
7. **Real-Time Predictions** – (Optional) Try it out on your own input and see how the model responds.

---

## 🛠️ Tech Stack

* 🐍 Python
* 🤗 HuggingFace Transformers (`distilbert-base-multilingual-cased`)
* 🧠 PyTorch
* 📊 Scikit-learn
* 📈 Matplotlib, Seaborn, Plotly
* 🚀 Google Colab

---

## 📁 Project Structure

```bash
Toxic-Comment-Classification-with-DistilBERT/
│
├── jigsaw-toxic-comment-train.csv         # Training dataset
├── validation.csv                         # Validation dataset
├── test.csv                               # Multilingual test data
├── test_labels.csv                        # Ground-truth test labels
├── tokenizer/                             # Tokenizer config files
├── toxic_model_v1.pt                      # Saved model weights
├── NLP Toxic Comment New.ipynb            # Main Colab notebook
├── README.md                              # You're here!
```

---

## 🚀 Getting Started

### Step 1: Clone the repository in Google Colab

```python
!git clone https://github.com/monal28/Toxic-Comment-Classification-with-DistilBERT.git
%cd Toxic-Comment-Classification-with-DistilBERT
```

### Step 2: Set up GPU and import dependencies

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

### Step 3: Load and preprocess the data

```python
import pandas as pd
train = pd.read_csv('jigsaw-toxic-comment-train.csv', nrows=10000)
valid = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv', nrows=5000)
```

### Step 4: Tokenization using DistilBERT

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
```

---

## 🏗️ Model Architecture

We fine-tune a **DistilBERT** transformer using a simple classification head (Linear Layer) that predicts whether a comment is toxic or not.

```python
class ToxicClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits
```

---

## 📈 Training & Metrics

* Optimizer: **Adam**
* Loss: **Weighted BCEWithLogitsLoss** (to handle class imbalance)
* Evaluation Metrics:

  * **Accuracy**
  * **ROC-AUC**
  * **Confusion Matrix**

---

## 🔍 Performance Snapshot

| Metric   | Multilingual | English-only |
| -------- | ------------ | ------------ |
| Accuracy | 81.88%       | 85.70%       |
| ROC-AUC  | 87.56%       | 94.27%       |

🔵 Confusion matrices and plots are available in the notebook.

---

## 🤔 Why DistilBERT?

DistilBERT is a lighter, faster alternative to BERT with **97% of its performance**, making it ideal for training in limited-resource environments like Colab.

🌐 Bonus: Using the *multilingual* version of DistilBERT lets us work with comments written in **multiple languages**, not just English.

---

## 💡 Future Improvements

* Add attention heatmaps to interpret model focus
* Incorporate ensemble with other transformer models
* Extend to multi-label classification for subtypes of toxicity

---

## **Web Deployement! 💻⚡**
The model is deployed on a web interface. You can test it live by scanning the QR code below or [clicking here](https://shaggy-lot-042991.framer.app).

![494360452_1354147595868795_387994747356468779_n](https://github.com/user-attachments/assets/ebefefdc-a92f-48b8-a212-c0f4218c9b16)

---
**Happy coding! 💻⚡**
*Let’s make the internet a little less toxic, one comment at a time.*

---
