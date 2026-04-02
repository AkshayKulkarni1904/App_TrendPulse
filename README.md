# App TrendPulse 🚀

AI-powered analytics platform for Google Play Store apps.

## Features
- Live data scraping from Google Play Store (15 apps, 200 reviews each)
- VADER sentiment analysis on user reviews
- Real monetization classification from Google Play metadata
- KMeans clustering with dynamic segment labeling
- Cross-validated RandomForest engagement prediction
- Scaled cosine similarity recommendation engine
- Rolling-average trend analysis
- Interactive Streamlit dashboard with dark/light theme toggle
- FastAPI backend serving data & ML predictions

## Project Structure

```
App_TrendPulse/
├── notebook/
│   ├── App_TrendPulse_Optimized.ipynb   # Main analytics notebook (Jupyter)
│   └── analytics_pipeline.py            # Importable pipeline module
├── api/
│   └── main.py                          # FastAPI backend
├── app/
│   ├── streamlit_app.py                 # Streamlit dashboard
│   ├── dataset.csv                      # Generated dataset
│   └── reviews_store.csv                # Customer review logs
├── assets/
│   └── logo.png                         # Dashboard logo
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data (Jupyter Notebook)
```bash
jupyter notebook notebook/App_TrendPulse_Optimized.ipynb
```
Run all cells to scrape Google Play data, train the ML model, and export `dataset.csv` + `model.pkl` to `app/`.

Alternatively, run the pipeline as a script:
```bash
cd notebook
python analytics_pipeline.py
```

### 3. Start the API Backend
```bash
cd api
uvicorn main:app --reload --port 8000
```
Wait for `Application startup complete` before proceeding.

### 4. Launch the Dashboard
In a **separate terminal**:
```bash
cd app
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/data` | Full dataset as JSON |
| POST | `/predict` | Predict engagement score for a campaign |
| POST | `/recommend` | Get similar apps via cosine similarity |
| POST | `/retrain` | Force ML model retrain from latest data |

## Tech Stack
- **Data**: pandas, numpy, google-play-scraper, vaderSentiment
- **ML**: scikit-learn (RandomForest, KMeans, cosine similarity)
- **Viz**: Plotly (interactive charts)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit