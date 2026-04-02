import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from functools import lru_cache

# ---- Configuration ----
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "dataset.csv")
REVIEWS_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "reviews_store.csv")


def fetch_live_dataset():
    """Read the latest dataset from disk. Called per-request to ensure live data."""
    resolved = os.path.abspath(DATA_PATH)
    if not os.path.exists(resolved):
        raise HTTPException(status_code=500, detail=f"Dataset not found at {resolved}")
    df = pd.read_csv(resolved).dropna(subset=["App"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def build_ml_pipeline():
    """Train the prediction pipeline once and cache it.
    Call build_ml_pipeline.cache_clear() to force retrain after data changes."""
    df = fetch_live_dataset()
    df["Monetization"] = df["Monetization"].fillna("Free")
    df["Genre"] = df["Genre"].fillna("Unknown")
    df["Prior_Sentiment"] = df.get("Sentiment", 0.0)

    X = df[["Installs", "Monetization", "Genre", "Prior_Sentiment"]]
    y = df["Engagement_Score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Installs", "Prior_Sentiment"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Monetization", "Genre"]),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X, y)
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the ML cache on startup so the first prediction is instant."""
    try:
        build_ml_pipeline()
    except Exception:
        pass
    yield


app = FastAPI(title="App TrendPulse API", version="2.0", lifespan=lifespan)


# ---- Pydantic Schemas ----
class PredictRequest(BaseModel):
    installs: float = Field(..., gt=0, description="Expected initial install count")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Target public sentiment (-1 to 1)")
    genre: str = Field(..., description="Target app genre")
    monetization: str = Field(..., description="Monetization strategy")


class RecommendRequest(BaseModel):
    app_name: str
    top_n: int = Field(default=5, ge=1, le=15)


# ---- Endpoints ----
@app.get("/")
def health_check():
    return {"status": "online", "version": "2.0"}


@app.get("/data")
def get_dataset():
    """Serve the full live dataset as JSON records."""
    df = fetch_live_dataset()
    return df.to_dict(orient="records")


@app.post("/predict")
def predict_engagement(request: PredictRequest):
    """Predict engagement score for a hypothetical app launch campaign."""
    try:
        pipeline = build_ml_pipeline()
        input_data = pd.DataFrame([{
            "Installs": request.installs,
            "Monetization": request.monetization,
            "Genre": request.genre,
            "Prior_Sentiment": request.sentiment,
        }])
        pred = pipeline.predict(input_data)[0]
        return {"predicted_engagement_score": round(float(pred), 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend_apps(request: RecommendRequest):
    """Find structurally similar apps using cosine similarity."""
    df = fetch_live_dataset()
    features = df[["Rating", "Sentiment", "Engagement_Score"]].fillna(0)
    sim = cosine_similarity(features)

    if request.app_name not in df["App"].values:
        raise HTTPException(status_code=404, detail=f"'{request.app_name}' not found in dataset.")

    idx = df[df["App"] == request.app_name].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    recs = [df.iloc[s[0]]["App"] for s in scores[1 : request.top_n + 1]]
    return {"source": request.app_name, "recommendations": recs}


@app.post("/retrain")
def retrain_model():
    """Force the ML pipeline to retrain from the latest dataset on disk."""
    build_ml_pipeline.cache_clear()
    try:
        build_ml_pipeline()
        return {"status": "Model retrained successfully from latest data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))