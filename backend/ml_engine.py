"""Core ML/NLP services for SentimentIQ Pro.

The sentiment classifier is a TF-IDF + Logistic Regression pipeline. If an uploaded
CSV has a rating column, the classifier is trained and evaluated on that dataset.
Without labels, a small bundled demo corpus is used and the UI clearly marks it as
such. Topic discovery, anomaly detection, semantic retrieval and drift checks are
computed from the uploaded reviews.
"""
from __future__ import annotations
import re, math
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

LABELS = ["Negative", "Neutral", "Positive"]
RATING_COLS = {"overall", "rating", "stars", "score", "review_rating"}
TEXT_CANDIDATES = ["reviewtext", "review", "text", "content", "comment", "feedback", "summary"]
DATE_CANDIDATES = ["date", "reviewdate", "timestamp", "created_at", "time"]
GROUP_CANDIDATES = ["product", "productname", "brand", "category", "item", "asin"]

DEMO_TEXTS = [
("excellent amazing wonderful love fantastic perfect great highly recommend", "Positive"),
("works beautifully smooth fast easy to use worth the money", "Positive"),
("very happy with quality and performance exceeded expectations", "Positive"),
("good product reliable delivery and helpful support", "Positive"),
("the graphics are beautiful and gameplay is engaging", "Positive"),
("terrible awful disappointing broken poor waste of money", "Negative"),
("stopped working crashes frequently defective damaged", "Negative"),
("late delivery missing parts bad quality frustrating", "Negative"),
("customer support was rude and never solved my issue", "Negative"),
("slow loading bugs and confusing controls", "Negative"),
("it is okay average nothing special", "Neutral"),
("works as expected neither good nor bad", "Neutral"),
("decent quality acceptable price and ordinary performance", "Neutral"),
("the product is fine but could be better", "Neutral"),
("mixed experience some good parts and some problems", "Neutral"),
]

ASPECTS = {
    "Product quality": ["quality", "material", "build", "durable", "broken", "defective", "damaged", "sturdy"],
    "Delivery": ["delivery", "shipping", "arrived", "late", "package", "packaging", "courier"],
    "Price & value": ["price", "cost", "expensive", "cheap", "value", "money", "worth"],
    "Performance": ["performance", "speed", "slow", "fast", "crash", "crashes", "bug", "loading", "battery"],
    "Customer support": ["support", "service", "customer", "refund", "response", "help", "staff"],
    "Usability": ["controls", "easy", "difficult", "confusing", "interface", "setup", "use"],
    "Features & experience": ["graphics", "gameplay", "soundtrack", "feature", "design", "story", "experience"],
}
POS_WORDS = set("excellent amazing wonderful love fantastic perfect great happy smooth fast easy reliable beautiful enjoyable recommend good best awesome satisfied works exceeded".split())
NEG_WORDS = set("terrible awful disappointing broken poor waste stopped crashes defective damaged late missing bad frustrating rude slow bugs confusing worst issue regret".split())


def norm_col(c: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if norm_col(candidate) in lookup:
            return lookup[norm_col(candidate)]
    return None


def infer_text_column(df: pd.DataFrame) -> str:
    col = find_column(df, TEXT_CANDIDATES)
    if col:
        return col
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("No review text column found. Include a column such as reviewText, review, text, or content.")
    return max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())


def rating_to_label(value: Any) -> str | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 2.49: return "Negative"
    if 2.5 <= v <= 3.49: return "Neutral"
    if 3.5 <= v <= 5: return "Positive"
    return None


def clean_text(value: Any) -> str:
    if pd.isna(value): return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000, sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    pipe.fit(texts, labels)
    return pipe


def safe_json_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return None
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    return x


def explain_prediction(pipe: Pipeline, text: str, predicted: str, top_n: int = 5) -> list[dict]:
    vec = pipe.named_steps["tfidf"].transform([text])
    clf = pipe.named_steps["classifier"]
    feature_names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    class_idx = list(clf.classes_).index(predicted)
    contributions = vec.toarray()[0] * clf.coef_[class_idx]
    indices = np.argsort(np.abs(contributions))[::-1]
    out = []
    for i in indices:
        if contributions[i] == 0: continue
        out.append({"term": str(feature_names[i]), "impact": round(float(contributions[i]), 4), "direction": "supports" if contributions[i] > 0 else "opposes"})
        if len(out) >= top_n: break
    return out


def aspect_analysis(text: str) -> list[dict]:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    results = []
    for aspect, terms in ASPECTS.items():
        hits = [t for t in terms if t in tokens]
        if not hits: continue
        pos = len(tokens & POS_WORDS); neg = len(tokens & NEG_WORDS)
        # A transparent local heuristic: sentiment words in the same review are attributed to mentioned aspects.
        sentiment = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
        results.append({"aspect": aspect, "sentiment": sentiment, "matched_terms": hits})
    return results


def topic_discovery(texts: list[str], n_topics: int = 5) -> list[dict]:
    valid = [t for t in texts if len(t.split()) >= 2]
    if len(valid) < 4: return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2500, min_df=1, ngram_range=(1, 2))
        X = vectorizer.fit_transform(valid)
        k = min(n_topics, max(2, len(valid) // 3), X.shape[1])
        if X.shape[1] < 2: return []
        model = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=300)
        W = model.fit_transform(X)
        terms = vectorizer.get_feature_names_out()
        topics = []
        for i, row in enumerate(model.components_):
            top = row.argsort()[::-1][:6]
            members = np.argsort(W[:, i])[::-1][:3]
            topics.append({"name": f"Topic {i+1}", "keywords": [str(terms[j]) for j in top], "review_count": int((W[:, i] > np.percentile(W[:, i], 65)).sum()), "examples": [valid[j][:180] for j in members]})
        return topics
    except Exception:
        return []


def drift_analysis(texts: list[str]) -> dict:
    if len(texts) < 10:
        return {"available": False, "message": "At least 10 reviews are needed to compare the first and second halves of the upload."}
    mid = len(texts) // 2
    a, b = texts[:mid], texts[mid:]
    v = TfidfVectorizer(max_features=1500, stop_words="english")
    try:
        X = v.fit_transform(a + b)
        A = np.asarray(X[:len(a)].mean(axis=0)).ravel() + 1e-9
        B = np.asarray(X[len(a):].mean(axis=0)).ravel() + 1e-9
        A, B = A / A.sum(), B / B.sum()
        m = (A + B) / 2
        js = 0.5 * np.sum(A * np.log(A / m)) + 0.5 * np.sum(B * np.log(B / m))
        return {"available": True, "score": round(float(js), 4), "level": "High" if js > .15 else "Moderate" if js > .06 else "Low", "message": "Vocabulary distribution comparison between the first and second halves of the uploaded file; this is a lightweight drift indicator, not a production drift guarantee."}
    except Exception as e:
        return {"available": False, "message": f"Could not calculate drift: {e}"}


def build_analysis(df: pd.DataFrame) -> dict:
    if df.empty: raise ValueError("The uploaded CSV contains no rows.")
    text_col = infer_text_column(df)
    work = df.copy()
    work["__review"] = work[text_col].map(clean_text)
    work = work[work["__review"].str.len() > 0].reset_index(drop=True)
    if work.empty: raise ValueError("No non-empty review text was found in the selected column.")
    texts = work["__review"].tolist()
    rating_col = find_column(work, list(RATING_COLS))
    labels = [rating_to_label(v) for v in work[rating_col]] if rating_col else [None] * len(work)
    labelled = [(t, l) for t, l in zip(texts, labels) if l]
    metrics = {"available": False, "note": "Upload a rating column (1–5) to train and evaluate on your own labels."}
    if len(labelled) >= 9 and len(set(l for _, l in labelled)) >= 2:
        X_text = [x[0] for x in labelled]; y = [x[1] for x in labelled]
        counts = Counter(y)
        stratify = y if min(counts.values()) >= 2 else None
        try:
            xtr, xte, ytr, yte = train_test_split(X_text, y, test_size=max(0.2, 3/len(y)), random_state=42, stratify=stratify)
        except ValueError:
            xtr, xte, ytr, yte = train_test_split(X_text, y, test_size=.25, random_state=42)
        eval_model = train_classifier(xtr, ytr)
        yp = eval_model.predict(xte)
        p, r, f, _ = precision_recall_fscore_support(yte, yp, labels=LABELS, average="weighted", zero_division=0)
        metrics = {"available": True, "training_rows": len(xtr), "test_rows": len(xte), "accuracy": round(float(accuracy_score(yte, yp))*100, 2), "precision": round(float(p)*100, 2), "recall": round(float(r)*100, 2), "f1": round(float(f)*100, 2), "labels": LABELS, "confusion_matrix": confusion_matrix(yte, yp, labels=LABELS).tolist(), "note": "Metrics are from a stratified holdout where possible. Small datasets can produce unstable estimates."}
        model = train_classifier(X_text, y)
        model_source = "Trained on uploaded ratings"
    else:
        X_text = [x for x, _ in DEMO_TEXTS]; y = [l for _, l in DEMO_TEXTS]
        model = train_classifier(X_text, y)
        model_source = "Bundled demo model (no usable rating labels detected)"
    pred = model.predict(texts)
    probs = model.predict_proba(texts)
    class_order = list(model.named_steps["classifier"].classes_)
    confidence = probs.max(axis=1)
    records = []
    for i, (txt, sent, conf, prob) in enumerate(zip(texts, pred, confidence, probs)):
        records.append({"index": int(i), "review": txt[:1000], "sentiment": str(sent), "confidence": round(float(conf)*100, 1), "uncertain": bool(conf < .55), "probabilities": {str(c): round(float(prob[j])*100, 1) for j, c in enumerate(class_order)}, "explanation": explain_prediction(model, txt, str(sent))})
    counts = Counter(pred)
    total = len(records)
    sentiment_counts = {s: int(counts.get(s, 0)) for s in LABELS}
    sentiment_pct = {s: round(sentiment_counts[s] / total * 100, 1) for s in LABELS}
    neg_texts = [r["review"] for r in records if r["sentiment"] == "Negative"]
    keyword_counts = Counter()
    for t in neg_texts:
        keyword_counts.update(w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in {"the","and","for","with","this","that","was","but","not","are","you","very","have","has","from","they","too","its"})
    issues = [{"keyword": k, "count": int(v)} for k,v in keyword_counts.most_common(10)]
    # Anomaly detection over text length, confidence, and duplicate frequency.
    lengths = np.array([len(t.split()) for t in texts], dtype=float)
    dup = pd.Series(texts).map(pd.Series(texts).value_counts()).to_numpy(dtype=float)
    features = np.column_stack([lengths, confidence, dup])
    if len(texts) >= 8:
        anomaly_model = IsolationForest(contamination="auto", random_state=42)
        anomaly_flags = anomaly_model.fit_predict(features) == -1
    else:
        anomaly_flags = np.zeros(len(texts), dtype=bool)
    anomalies = [{"review": records[i]["review"], "reason": "Unusual combination of review length, model confidence, or repetition frequency", "sentiment": records[i]["sentiment"]} for i in np.where(anomaly_flags)[0][:30]]
    # Aspect-based results
    aspect_counts = {a: Counter() for a in ASPECTS}
    for rec in records:
        for aspect in aspect_analysis(rec["review"]): aspect_counts[aspect["aspect"]][aspect["sentiment"]] += 1
    aspects = [{"aspect": a, "positive": int(c["Positive"]), "neutral": int(c["Neutral"]), "negative": int(c["Negative"]), "mentions": int(sum(c.values()))} for a,c in aspect_counts.items() if sum(c.values())]
    # Forecasting only if date/time column parses and enough observations
    date_col = find_column(work, DATE_CANDIDATES)
    forecast = {"available": False, "message": "Add a date/reviewDate/timestamp column with parseable dates to enable sentiment trend forecasting."}
    if date_col:
        dates = pd.to_datetime(work[date_col], errors="coerce")
        valid = pd.DataFrame({"date": dates, "negative": [1 if x == "Negative" else 0 for x in pred]}).dropna()
        if len(valid) >= 8:
            daily = valid.set_index("date").resample("W").negative.mean().dropna()
            if len(daily) >= 4:
                x = np.arange(len(daily)); coef = np.polyfit(x, daily.values, 1)
                future_x = np.arange(len(daily), len(daily)+4)
                vals = np.clip(np.polyval(coef, future_x), 0, 1)
                forecast = {"available": True, "method": "Linear trend baseline", "historical": [{"period": str(d.date()), "negative_rate": round(float(v)*100,1)} for d,v in daily.items()], "forecast": [{"period": f"Week +{i+1}", "negative_rate": round(float(v)*100,1)} for i,v in enumerate(vals)], "message": "Exploratory four-week trend extrapolation. It is not a validated forecasting model."}
            else: forecast = {"available": False, "message": "Not enough weekly observations after date aggregation; at least four weeks are needed."}
    # Label evaluation and class balance details
    topic = topic_discovery(texts)
    drift = drift_analysis(texts)
    insights = []
    if sentiment_pct["Negative"] >= 35: insights.append("Negative sentiment is elevated; investigate the most frequent issue keywords.")
    if sentiment_pct["Positive"] >= 65: insights.append("Positive feedback dominates this upload; identify the recurring strengths to preserve.")
    if any(r["uncertain"] for r in records): insights.append(f"{sum(r['uncertain'] for r in records)} reviews have low model confidence and should be reviewed by a person.")
    if not insights: insights.append("Sentiment is mixed. Use aspect mentions and topic clusters to locate specific improvement opportunities.")
    # Comparison groups
    group_col = find_column(work, GROUP_CANDIDATES)
    groups = []
    if group_col:
        temp = pd.DataFrame({"group": work[group_col].astype(str), "sentiment": pred})
        for name, g in temp.groupby("group"):
            if len(g) < 1: continue
            groups.append({"name": name, "reviews": int(len(g)), "positive_pct": round(float((g.sentiment=="Positive").mean())*100,1), "neutral_pct": round(float((g.sentiment=="Neutral").mean())*100,1), "negative_pct": round(float((g.sentiment=="Negative").mean())*100,1)})
        groups = sorted(groups, key=lambda x: x["reviews"], reverse=True)[:20]
    return {
        "summary": {"total": total, "counts": sentiment_counts, "percentages": sentiment_pct, "satisfaction": sentiment_pct["Positive"], "model_source": model_source, "text_column": str(text_col), "rating_column": str(rating_col) if rating_col else None, "date_column": str(date_col) if date_col else None, "group_column": str(group_col) if group_col else None},
        "reviews": records, "issues": issues, "insights": insights, "topics": topic, "aspects": aspects,
        "anomalies": anomalies, "metrics": metrics, "drift": drift, "forecast": forecast, "groups": groups,
        "model_classes": class_order
    }


def semantic_search(reviews: list[dict], query: str, limit: int = 8) -> list[dict]:
    if not query.strip(): return []
    texts = [r["review"] for r in reviews]
    if not texts: return []
    v = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    try:
        X = v.fit_transform(texts + [query])
        scores = cosine_similarity(X[-1], X[:-1]).ravel()
        idx = np.argsort(scores)[::-1][:limit]
        return [{"review": reviews[i]["review"], "sentiment": reviews[i]["sentiment"], "confidence": reviews[i]["confidence"], "similarity": round(float(scores[i])*100,1)} for i in idx if scores[i] > 0]
    except ValueError: return []


def answer_question(reviews: list[dict], question: str) -> dict:
    q = question.lower()
    if not reviews: return {"answer": "Upload a review CSV first.", "evidence": []}
    if any(w in q for w in ["complain", "problem", "issue", "negative", "bad"]):
        selected = [r for r in reviews if r["sentiment"] == "Negative"]
        intro = f"I found {len(selected)} negative reviews ({len(selected)/len(reviews)*100:.1f}% of the uploaded reviews)."
    elif any(w in q for w in ["like", "love", "positive", "good", "strength"]):
        selected = [r for r in reviews if r["sentiment"] == "Positive"]
        intro = f"I found {len(selected)} positive reviews ({len(selected)/len(reviews)*100:.1f}% of the uploaded reviews)."
    else:
        selected = semantic_search(reviews, question, 5)
        evidence = selected[:5]
        return {"answer": "Here are the reviews most relevant to your question. This assistant retrieves evidence from your upload; it does not generate unsupported facts.", "evidence": evidence}
    ranked = semantic_search(selected, question, 5) if selected else []
    return {"answer": intro + " The examples below are the closest matching reviews in your dataset.", "evidence": ranked or selected[:5]}
