# Toxic-Comment-Classification-with-DistilBERT
Multilingual toxic comment classifier using DistilBERT. Trained on Jigsaw dataset for binary classification (toxic vs non-toxic), with support for evaluation on multilingual and English-only data. Includes training, validation, prediction, and confusion matrix visualisation.

This project uses the distilbert-base-multilingual-cased model to detect toxic comments in multiple languages. It fine-tunes a transformer-based classifier on the Jigsaw dataset for binary classification (toxic vs. non-toxic), with special handling for class imbalance and multilingual evaluation.

🔧 Key Features
- Multilingual Support: Trained on multilingual text and tested on both multilingual and English-only datasets.

- Model Architecture: Based on DistilBERT with a linear classification head.

- Class Imbalance Handling: Uses BCEWithLogitsLoss with pos_weight to penalize toxic comments appropriately.

- Performance Metrics: Tracks Accuracy, AUC, and displays confusion matrices for evaluation.

- Inference Support: Includes a function to predict toxicity of personal input texts.

- Efficient Processing: Tokenization with batching and chunking for scalability.

📁 Structure
- data/: Contains input datasets.

- model/: DistilBERT classifier definition.

- train.py: Training loop with metrics.

- evaluate.py: Validation and multilingual/English evaluation.

- inference.py: Predict single comment toxicity.

- toxic_model_v1.pt: Saved model weights.

- tokenizer/: Pretrained tokenizer files.

📊 Dataset - Jigsaw Multilingual Toxic Comment Classification

✅ Results
Achieves strong accuracy and AUC on both multilingual and English test sets. Confusion matrices show balanced detection across toxic and non-toxic labels.

💡 Future Improvements
- Fine-tune on larger transformer models (e.g., XLM-R)

- Try ensemble methods for better generalization

- Deploy as an API for real-time toxicity detection
