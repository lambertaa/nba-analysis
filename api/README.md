# NBA Draft All-Star Prediction API

A production-ready FastAPI application that predicts whether an NBA draft prospect will become an All-Star based on scouting reports using NLP and machine learning.

## Features

- **Two Prediction Endpoints:**
  1. `/predict/by-name` - Scrapes scouting report from nbadraft.net by player name
  2. `/predict/by-report` - Accepts manual scouting report input

- **NLP-Powered:** Uses TF-IDF vectorization with separate processing for strengths and weaknesses
- **Interpretable:** Returns top features contributing to each prediction
- **Dockerized:** Easy deployment with Docker and Docker Compose
- **Health Checks:** Built-in health monitoring endpoints

## Quick Start

### 1. Export Model from MLflow

First, export your trained model artifacts:

```bash
cd api
python export_model.py
```

This will create a `models/` directory with:
- `model.pkl` - Trained logistic regression model
- `vectorizer_strengths.pkl` - TF-IDF vectorizer for strengths
- `vectorizer_weaknesses.pkl` - TF-IDF vectorizer for weaknesses
- `optimal_threshold.txt` - Optimal decision threshold
- `feature_names.txt` - Feature names for interpretability

### 2. Run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### 3. Run Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns model status and information.

### Predict by Player Name

```bash
POST /predict/by-name
Content-Type: application/json

{
  "player_name": "Cooper Flagg"
}
```

### Predict by Scouting Report

```bash
POST /predict/by-report
Content-Type: application/json

{
  "strengths": "Elite three point shooter with deep range...",
  "weaknesses": "Average lateral quickness on defense...",
  "overall": 88.0,
  "athleticism": 75.0,
  "size": 80.0,
  "defense": 70.0,
  "rebounding": 65.0,
  "jump_shot": 92.0,
  "nba_ready": 85.0
}
```

### Response Format

```json
{
  "player_name": "Cooper Flagg",
  "prediction": "All-Star",
  "probability": 0.82,
  "confidence": "High",
  "threshold_used": 0.65,
  "top_positive_features": {
    "STR_elite three point": 0.85,
    "STR_high basketball iq": 0.72
  },
  "top_negative_features": {
    "WEAK_average lateral quickness": -0.45
  }
}
```

## Testing

Run example requests:

```bash
python example_requests.py
```

## Interactive API Documentation

Once the API is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
api/
├── app.py                  # FastAPI application
├── predictor.py            # Model loading and prediction logic
├── scraper.py              # Web scraping for nbadraft.net
├── export_model.py         # Export model from MLflow
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── example_requests.py     # Example API requests
└── models/                 # Model artifacts (created by export_model.py)
    ├── model.pkl
    ├── vectorizer_strengths.pkl
    ├── vectorizer_weaknesses.pkl
    ├── optimal_threshold.txt
    └── feature_names.txt
```

## Model Information

- **Algorithm:** Logistic Regression with L2 regularization
- **Features:** TF-IDF vectors (n-grams 2-4) + numerical grades
- **Target:** All-Star within 7 years of draft
- **Threshold:** Optimized for F1 score (~0.65-0.80)
- **Training Data:** ~400-500 NBA draft prospects with scouting reports

## Environment Variables

- `LOG_LEVEL` - Logging level (default: INFO)

## Notes

- The web scraper may not work for all players if nbadraft.net structure changes
- Numerical grades are optional; the model can work with text-only input
- The model uses the same preprocessing pipeline as training (stopword removal, lemmatization)

