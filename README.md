# Fake Job Detector — India Edition

A machine learning system that detects fraudulent job postings targeting Indian job seekers.

---

## Problem

Fake job scams in India exploit platforms like Naukri, LinkedIn, and Indeed.
Victims are tricked into paying registration fees, sharing Aadhar/PAN details,
or joining fake work-from-home schemes. This tool flags suspicious postings
before job seekers fall for them.

---

## What It Does

- Detects fraud using **42 hand-engineered features** + TF-IDF text analysis
- Identifies **India-specific scam patterns**: portal impersonation, Aadhar/UPI requests, WhatsApp contacts, fee language
- Provides **real-time fraud scoring** via a Streamlit web app
- Explains exactly *why* a posting is flagged (red flag breakdown)

---

## Models Trained

| Model | F1 Score |
|---|---|
| Random Forest (baseline) | ~0.90 |
| LSTM | ~0.91 |
| **XGBoost (final model)** | **~0.93** |

XGBoost was chosen as the final model for its best F1 score and faster inference.

---

## Dataset

- **Source**: EMSI Fake Job Postings dataset (Kaggle)
- **Size**: 17,880 job postings
- **Class imbalance**: ~4.8% fraudulent — handled using SMOTE + `scale_pos_weight`
- India-specific fraud signals were added as hand-engineered features on top of this dataset

---

## Project Structure

```
fake-job-detector/
├── model_training.ipynb     # Full ML pipeline: EDA → preprocessing → training → evaluation
├── streamlit_app.py         # Live fraud detection web app
├── scraper.py               # Web scraper for Naukri/Indeed (educational use only)
├── fake_job_postings.csv    # Training dataset
├── requirements.txt         # Python dependencies
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

Also download NLTK data (run once):
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Step 2 — Train the model
Open and run `model_training.ipynb` top to bottom in Jupyter.

This will create:
```
models/
├── xgboost_model.pkl
├── random_forest.pkl
├── lstm_model.h5
└── feature_names.pkl

processed_data/
├── tfidf_vectorizer.pkl
└── scaler.pkl

plots/
├── confusion_matrix_XGBoost.png
├── feature_importance.png
└── model_comparison.png
```

### Step 3 — Launch the web app
```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## Key Features Engineered

| Category | Features |
|---|---|
| Text statistics | word count, sentence count, avg word length |
| Punctuation abuse | exclamation count, caps ratio |
| India fraud signals | portal impersonation, WhatsApp contact, fee language |
| Document requests | Aadhar, PAN, UPI, bank details |
| Sentiment | VADER compound, positive, negative scores |
| Salary analysis | missing salary, unrealistic salary for freshers |
| Structural | missing company info, vague job title, missing requirements |

---

## Web App Screenshot

Enter any job posting details → get an instant fraud probability score + red flag breakdown.

---

## Limitations

- Training data is a global dataset (not India-specific raw data). India fraud detection relies on hand-engineered rules.
- Scraper (`scraper.py`) may not work if Naukri/Indeed update their HTML structure.
- Model cannot analyze images or attachments in job postings.

---

## Built By

**Divya Virkud** | v2.0 | 2026  
For educational purposes only.
