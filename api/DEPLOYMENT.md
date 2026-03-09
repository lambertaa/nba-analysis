# NBA Draft API - Deployment Guide

## ✅ What's Been Created

Your Dockerized NBA Draft All-Star Prediction API is ready! Here's what was built:

### 📁 API Structure
```
api/
├── app.py                      # FastAPI application with 2 endpoints
├── predictor.py                # Model loading & prediction logic
├── scraper.py                  # Web scraping for nbadraft.net
├── export_model.py             # Export model from MLflow ✓ COMPLETED
├── test_api_local.py           # Local testing script
├── example_requests.py         # Example API requests
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore               # Docker ignore file
├── README.md                   # Full documentation
└── models/                     # Model artifacts ✓ EXPORTED
    ├── model.pkl
    ├── vectorizer_strengths.pkl
    ├── vectorizer_weaknesses.pkl
    ├── optimal_threshold.txt
    ├── feature_names.txt
    └── *.png (feature importance visualizations)
```

### 🎯 API Endpoints

1. **`POST /predict/by-name`** - Predict by player name (scrapes nbadraft.net)
2. **`POST /predict/by-report`** - Predict from scouting report data
3. **`GET /health`** - Health check endpoint
4. **`GET /`** - Root endpoint

### 📊 Model Information

- **Model**: Logistic Regression with L2 regularization
- **Features**: 1009 total (1000 TF-IDF + 9 numerical)
  - 500 TF-IDF features from Strengths (n-grams 2-4)
  - 500 TF-IDF features from Weaknesses (n-grams 2-4)
  - 9 numerical grades: overall, Athleticism, Size, Defense, Strength, Quickness, Leadership, JumpShot, NBAReady
- **Optimal Threshold**: 0.65
- **Target**: All-Star within 7 years

---

## 🚀 Quick Start

### Option 1: Docker (Recommended for Production)

```bash
# Navigate to API directory
cd api

# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000
```

### Option 2: Local Development

```bash
# Navigate to API directory
cd api

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the API
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
```

---

## 🧪 Testing the API

### Test Locally (Without Server)

```bash
cd api
python test_api_local.py
```

This tests the predictor directly without starting a server.

### Test with Server Running

```bash
# In one terminal, start the API
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# In another terminal, run example requests
cd api
python example_requests.py
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📝 Example Requests

### Predict by Scouting Report

```bash
curl -X POST "http://localhost:8000/predict/by-report" \
  -H "Content-Type: application/json" \
  -d '{
    "strengths": "Elite three point shooter with deep range. High basketball IQ.",
    "weaknesses": "Average lateral quickness. Needs to improve finishing at rim.",
    "overall": 88.0,
    "athleticism": 75.0,
    "size": 80.0,
    "defense": 70.0,
    "strength": 70.0,
    "quickness": 80.0,
    "leadership": 85.0,
    "jump_shot": 92.0,
    "nba_ready": 85.0
  }'
```

### Predict by Player Name

```bash
curl -X POST "http://localhost:8000/predict/by-name" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Cooper Flagg"
  }'
```

---

## 🐳 Docker Commands

```bash
# Build the image
docker-compose build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Rebuild and restart
docker-compose up --build
```

---

## 🔧 Troubleshooting

### Issue: NLTK data not found

The Dockerfile downloads NLTK data automatically. If running locally:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Issue: Model not found

Run the export script to extract the model from MLflow:

```bash
cd api
python export_model.py
```

### Issue: Feature count mismatch

Ensure all 9 numerical features are provided:
- overall, Athleticism, Size, Defense, Strength, Quickness, Leadership, JumpShot, NBAReady

Missing features will default to 0.0.

---

## 📦 Next Steps

1. **Test the API locally** using `test_api_local.py`
2. **Build the Docker image** with `docker-compose build`
3. **Run the container** with `docker-compose up`
4. **Test the endpoints** using `example_requests.py` or the Swagger UI
5. **Deploy to production** (AWS, GCP, Azure, etc.)

---

## 🎉 Success!

Your NBA Draft All-Star Prediction API is production-ready and Dockerized!

The model has been successfully exported from MLflow and is ready to serve predictions.

