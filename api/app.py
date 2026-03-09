"""
NBA Draft All-Star Prediction API

FastAPI application that provides endpoints for predicting All-Star potential
of NBA draft prospects using NLP on scouting reports.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import logging

from predictor import NBADraftPredictor
from scraper import scrape_player_scouting_report

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Draft All-Star Prediction API",
    description="Predict All-Star potential of NBA draft prospects using NLP on scouting reports",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loads model on startup)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    logger.info("Loading NBA Draft prediction model...")
    try:
        predictor = NBADraftPredictor()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Request/Response Models
class PlayerNameRequest(BaseModel):
    """Request model for player name lookup"""

    player_name: str = Field(..., description="Player name to look up on nbadraft.net")

    class Config:
        schema_extra = {"example": {"player_name": "Cooper Flagg"}}


class ScoutingReportRequest(BaseModel):
    """Request model for direct scouting report input"""

    strengths: str = Field(..., description="Player's strengths from scouting report")
    weaknesses: str = Field(..., description="Player's weaknesses from scouting report")
    overall: Optional[float] = Field(None, description="Overall draft grade (0-100)")
    athleticism: Optional[float] = Field(None, description="Athleticism grade (0-100)")
    size: Optional[float] = Field(None, description="Size grade (0-100)")
    defense: Optional[float] = Field(None, description="Defense grade (0-100)")
    strength: Optional[float] = Field(None, description="Strength grade (0-100)")
    quickness: Optional[float] = Field(None, description="Quickness grade (0-100)")
    leadership: Optional[float] = Field(None, description="Leadership grade (0-100)")
    jump_shot: Optional[float] = Field(None, description="Jump shot grade (0-100)")
    nba_ready: Optional[float] = Field(None, description="NBA readiness grade (0-100)")

    class Config:
        schema_extra = {
            "example": {
                "strengths": "Elite athleticism, great defensive potential, high basketball IQ",
                "weaknesses": "Needs to improve shooting range, limited offensive repertoire",
                "overall": 85.0,
                "athleticism": 90.0,
                "size": 80.0,
                "defense": 85.0,
                "strength": 75.0,
                "quickness": 85.0,
                "leadership": 70.0,
                "jump_shot": 65.0,
                "nba_ready": 70.0,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    player_name: Optional[str] = None
    prediction: str = Field(..., description="All-Star or Non-All-Star")
    probability: float = Field(
        ..., description="Probability of becoming an All-Star (0-1)"
    )
    confidence: str = Field(..., description="Confidence level: High, Medium, or Low")
    threshold_used: float = Field(
        ..., description="Decision threshold used for classification"
    )
    top_positive_features: Dict[str, float] = Field(
        ..., description="Top features predicting All-Star"
    )
    top_negative_features: Dict[str, float] = Field(
        ..., description="Top features predicting Non-All-Star"
    )

    class Config:
        schema_extra = {
            "example": {
                "player_name": "Cooper Flagg",
                "prediction": "All-Star",
                "probability": 0.82,
                "confidence": "High",
                "threshold_used": 0.65,
                "top_positive_features": {
                    "STR_elite athleticism": 0.85,
                    "STR_defensive potential": 0.72,
                },
                "top_negative_features": {"WEAK_shooting range": -0.45},
            }
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "NBA Draft All-Star Prediction API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_info": predictor.get_model_info() if predictor else None,
    }


@app.post("/predict/by-name", response_model=PredictionResponse)
async def predict_by_name(request: PlayerNameRequest):
    """
    Predict All-Star potential by player name.
    Scrapes scouting report from nbadraft.net.
    """
    try:
        logger.info(f"Received prediction request for player: {request.player_name}")

        # Scrape scouting report
        scouting_data = scrape_player_scouting_report(request.player_name)

        if not scouting_data:
            raise HTTPException(
                status_code=404,
                detail=f"Could not find scouting report for player: {request.player_name}",
            )

        # Make prediction
        prediction = predictor.predict(scouting_data)
        prediction["player_name"] = request.player_name

        logger.info(
            f"Prediction for {request.player_name}: {prediction['prediction']} ({prediction['probability']:.2%})"
        )

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting for {request.player_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/by-report", response_model=PredictionResponse)
async def predict_by_report(request: ScoutingReportRequest):
    """
    Predict All-Star potential from scouting report data.
    Accepts strengths, weaknesses, and numerical grades.
    """
    try:
        logger.info("Received prediction request with scouting report")

        # Convert request to dict format expected by predictor
        scouting_data = {
            "Strengths": request.strengths,
            "Weaknesses": request.weaknesses,
            "overall": request.overall,
            "Athleticism": request.athleticism,
            "Size": request.size,
            "Defense": request.defense,
            "Strength": request.strength,
            "Quickness": request.quickness,
            "Leadership": request.leadership,
            "JumpShot": request.jump_shot,
            "NBAReady": request.nba_ready,
        }

        # Make prediction
        prediction = predictor.predict(scouting_data)

        logger.info(
            f"Prediction: {prediction['prediction']} ({prediction['probability']:.2%})"
        )

        return prediction

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
