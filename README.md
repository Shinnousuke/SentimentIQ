# SentimentIQ Pro — AI-powered Customer Intelligence

A runnable full-stack project with an HTML/CSS/JavaScript dashboard and a FastAPI + scikit-learn backend. It expands the original three-class sentiment dashboard with explainability, unsupervised topic discovery, aspect-level insights, anomaly detection, semantic search, review Q&A, holdout evaluation, trend extrapolation, drift indicators, and a human feedback loop.

## Quick start (Windows)

1. Install **Python 3.10–3.12**.
2. Extract the ZIP and open the `SentimentIQ_Pro` folder in Terminal / PowerShell.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

4. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Start the app from the project root:

```powershell
python -m uvicorn backend.main:app --reload
```

6. Open **http://127.0.0.1:8000** in your browser. Keep the terminal open while using the app. Press `Ctrl+C` to stop it.

## Try it immediately

Upload `data/SentimentIQ_sample_reviews.csv` from the extracted project. It includes:

- `reviewText`: review text
- `overall`: 1–5 rating, mapped to Negative (1–2), Neutral (3), Positive (4–5)
- `reviewDate`: date for exploratory trend analysis
- `product`: group for product comparison

## Features included

| Feature | Implementation in this project | Data requirements / notes |
|---|---|---|
| Sentiment classification | TF-IDF word and bigram features + Logistic Regression | Trains on uploaded ratings when enough usable labels exist; otherwise uses a small bundled demo corpus, clearly labelled in the UI |
| Explainable AI | Signed Logistic Regression feature contributions for each prediction | Local model explanation; not SHAP/LIME |
| Topic discovery | Non-negative Matrix Factorization (NMF) over TF-IDF | Works best with several varied reviews |
| Aspect sentiment | Aspect dictionary + local sentiment-word heuristic | Lightweight ABSA prototype, not a fine-tuned transformer ABSA model |
| Anomaly detection | Isolation Forest on review length, confidence and repetition | Exploratory unusual-record detector; flagged items need human review |
| Semantic search | TF-IDF cosine similarity | Searches by overlapping meaning-bearing terms, not a transformer embedding model |
| Ask Your Reviews | Retrieval-based answer with matching review evidence | Not a generative LLM; does not invent unsupported answers |
| Model performance | Holdout accuracy, precision, recall, F1 and confusion matrix | Requires at least 9 usable labelled reviews and at least 2 classes; small datasets make metrics unstable |
| Sentiment trend forecast | Weekly negative-rate aggregation + linear trend extrapolation | Requires a parseable date/reviewDate/timestamp column and enough weekly observations; exploratory baseline only |
| Data drift | Jensen–Shannon divergence between TF-IDF vocabulary distributions in the first and second halves of the upload | Lightweight within-file drift signal, not production monitoring |
| Product comparison | Sentiment breakdown by product/brand/category-like column | Include a column named `product`, `brand`, `category`, `item`, etc. |
| Human feedback loop | Corrections stored in local SQLite; export to CSV | Corrections are not automatically fed back into the active model; retraining is a separate step |
| CSV export | Download row-level predictions and feedback corrections | Available from dashboard buttons / links |

## Expected CSV format

At minimum, include one text column such as `reviewText`, `review`, `text`, `content`, or `comment`. The app can infer a likely text column if it has another name. Optional columns unlock more analysis:

- Rating: `overall`, `rating`, `stars`, `score` (numeric 1–5)
- Date: `date`, `reviewDate`, `timestamp`, `created_at`
- Group: `product`, `productName`, `brand`, `category`, `item`, `asin`

The upload limit is 25 MB. Empty review rows are ignored. No user-uploaded model pickle files are loaded.

## Project structure

```text
SentimentIQ_Pro/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes, SQLite feedback, frontend hosting
│   └── ml_engine.py         # Classification and ML/NLP analysis
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/
│   └── SentimentIQ_sample_reviews.csv
├── requirements.txt
└── README.md
```

## API routes

- `GET /api/health` — service status
- `POST /api/analyze` — upload CSV and return analysis
- `POST /api/search` — semantic review search
- `POST /api/ask` — evidence-based review question
- `POST /api/feedback` — save a corrected sentiment
- `GET /api/feedback/export` — export corrections
- `GET /api/export` — export predictions
- `GET /docs` — interactive FastAPI API documentation

## Connecting your own Colab-trained model later

The current project trains a compatible scikit-learn text classifier from the uploaded ratings. Your earlier Colab pipeline used a 3-class label mapping (0 Negative, 1 Neutral, 2 Positive) and TF-IDF. To replace the fallback / train-on-upload classifier with your saved model, export the complete fitted preprocessing + classifier pipeline from Colab and add a controlled model-loading path in `backend/ml_engine.py`. Keep the exact same vectorizer vocabulary, preprocessing, label mapping and class order. Do not load pickle/joblib files supplied by untrusted users.

## Important limitations

This is a working educational prototype, not a production-grade customer intelligence service. The bundled demo model is trained on a tiny illustrative corpus and should not be treated as accurate on arbitrary domains. For meaningful results, upload a sufficiently large, representative dataset with ratings. Aspect sentiment is heuristic; the review assistant is retrieval-based; forecasting and drift are exploratory baselines. The app processes one active upload in memory per server process, and feedback is stored locally in SQLite.

## Troubleshooting

- **`python` not found:** install Python and enable “Add Python to PATH”.
- **Port 8000 busy:** run `python -m uvicorn backend.main:app --reload --port 8001` and open port 8001.
- **Module not found:** ensure the virtual environment is activated and run `pip install -r requirements.txt` from the project root.
- **API unavailable in page:** confirm the terminal shows Uvicorn running and open the exact local URL printed by it.
- **No metrics / forecast:** include usable rating/date columns; the dashboard intentionally reports when data is insufficient.

---

Built as an extensible applied NLP / ML project. Frontend uses Chart.js from CDN, so the browser needs internet access for the chart library and Google Fonts; the core API and analysis run locally.
