# 📊 Sentiment Analysis using BERT (Fine-Tuning with PyTorch & Hugging Face)

This repository contains a complete, beginner-friendly project for fine-tuning a pretrained **BERT-based model** for **binary sentiment analysis (Positive/Negative)** using **Hugging Face Transformers** and **PyTorch**, optimized for **low-resource environments like Google Colab**.

---

## 📁 Project Structure
```
├── train.csv                  # Training dataset (text, label)
├── dev.csv                   # Evaluation dataset (text, label)
├── sentiment_model/          # Saved fine-tuned model (after training)
├── training_metrics.csv      # Training metrics (loss & accuracy per epoch)
├── utils_plots.py            # Plotting utility functions (loss, accuracy, confusion matrix)
└── sentiment_training.ipynb  # Main training and evaluation notebook
```

---

## 🚀 Project Objectives
- Load and fine-tune a pretrained BERT model (or DistilBERT for low memory).
- Train on labeled sentiment data.
- Evaluate performance before and after training.
- Plot loss/accuracy curves and confusion matrix.
- Save model and metrics for reuse and reproducibility.

---

## 📦 Requirements
Install dependencies (preferably in Google Colab or a virtual environment):
```bash
pip install transformers datasets torch scikit-learn pandas tqdm seaborn matplotlib
```

---

## ⚙️ How to Use This Project

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/sentiment-bert-finetune.git
cd sentiment-bert-finetune
```

### 2. Prepare Data
Ensure `train.csv` and `dev.csv` have two columns:
```
text,label
"I love this product!",1
"Worst experience ever",0
```

### 3. Run Training
Use `sentiment_training.ipynb` in **Google Colab** for optimal GPU use.
- Choose BERT or DistilBERT based on memory.
- Configure training parameters (epochs, batch size, learning rate).
- Automatically saves model and tokenizer to `/content/drive/` if mounted.

### 4. Visualize Results
Use the `utils_plots.py` functions to:
- Plot loss and accuracy curves
- Show confusion matrix
- Export metrics to CSV

### 5. Evaluate Model
Evaluate on the dev set before and after training:
```python
from utils_plots import plot_confusion_matrix
plot_confusion_matrix(true_labels, predicted_labels)
```

### 6. Predict New Sentences
```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return "Positive" if pred == 1 else "Negative"
```

---

## 🛠 Key Hyperparameters (Adjustable)
| Parameter       | Description                                 | Suggested Value (Low Resource) |
|----------------|---------------------------------------------|-------------------------------|
| batch_size     | Number of samples per batch                 | 4-8                           |
| epochs         | Full passes through training set            | 2-3                           |
| learning_rate  | Speed of optimization                       | 2e-5 to 5e-5                  |
| max_length     | Max token length per sentence               | 128 or 256                    |

---

## 💡 Optimization Tips
- Use `DistilBERT` instead of `BERT` for faster training.
- Call `torch.cuda.empty_cache()` after each epoch.
- Use early stopping and monitor validation loss.
- Save your model to Google Drive to preserve work across sessions.

---

## 📈 Results & Evaluation
- Accuracy, loss curves, and confusion matrix help you analyze performance.
- All metrics are saved to CSV for easy reporting.

---

## ✨ Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

---

## 📜 License
MIT License. Use freely with attribution.

---

## 📬 Contact
For questions or collaborations, open an issue or email: `yourname@domain.com`

Happy fine-tuning! 🎯

