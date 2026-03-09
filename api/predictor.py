"""
NBA Draft Predictor Module

Handles model loading, preprocessing, and predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

# NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)


class NBADraftPredictor:
    """NBA Draft All-Star Prediction Model"""

    def __init__(self, model_path: str = "models"):
        """
        Initialize predictor and load model artifacts.

        Args:
            model_path: Path to directory containing model artifacts
        """
        self.model_path = model_path
        self.model = None
        self.vectorizer_strengths = None
        self.vectorizer_weaknesses = None
        self.feature_names = None
        self.optimal_threshold = 0.65  # Default from training
        self.numerical_features = [
            "overall",
            "Athleticism",
            "Size",
            "Defense",
            "Strength",
            "Quickness",
            "Leadership",
            "JumpShot",
            "NBAReady",
        ]

        # NLP preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Load model artifacts
        self._load_model()

    def _load_model(self):
        """Load model and vectorizers from disk"""
        try:
            # Load logistic regression model
            model_file = os.path.join(self.model_path, "model.pkl")
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {model_file}")

            # Load vectorizers
            str_vec_file = os.path.join(self.model_path, "vectorizer_strengths.pkl")
            weak_vec_file = os.path.join(self.model_path, "vectorizer_weaknesses.pkl")

            with open(str_vec_file, "rb") as f:
                self.vectorizer_strengths = pickle.load(f)
            logger.info(f"Loaded strengths vectorizer from {str_vec_file}")

            with open(weak_vec_file, "rb") as f:
                self.vectorizer_weaknesses = pickle.load(f)
            logger.info(f"Loaded weaknesses vectorizer from {weak_vec_file}")

            # Load optimal threshold if available
            threshold_file = os.path.join(self.model_path, "optimal_threshold.txt")
            if os.path.exists(threshold_file):
                with open(threshold_file, "r") as f:
                    self.optimal_threshold = float(f.read().strip())
                logger.info(f"Loaded optimal threshold: {self.optimal_threshold}")

            # Load feature names if available
            feature_file = os.path.join(self.model_path, "feature_names.txt")
            if os.path.exists(feature_file):
                with open(feature_file, "r") as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.feature_names)} feature names")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using same pipeline as training.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return " ".join(tokens)

    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction.

        Args:
            data: Dictionary containing scouting report data

        Returns:
            Feature array ready for model prediction
        """
        # Preprocess text
        strengths_clean = self.preprocess_text(data.get("Strengths", ""))
        weaknesses_clean = self.preprocess_text(data.get("Weaknesses", ""))

        # Vectorize text
        str_features = self.vectorizer_strengths.transform([strengths_clean])
        weak_features = self.vectorizer_weaknesses.transform([weaknesses_clean])

        # Prepare numerical features
        numerical_values = []
        for feat in self.numerical_features:
            value = data.get(feat)
            # Handle None/missing values
            numerical_values.append(value if value is not None else 0.0)

        numerical_array = np.array(numerical_values).reshape(1, -1)

        # Combine features (text + numerical)
        from scipy.sparse import hstack

        combined_features = hstack([str_features, weak_features, numerical_array])

        return combined_features

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a player.

        Args:
            data: Dictionary containing scouting report data

        Returns:
            Dictionary with prediction results
        """
        # Prepare features
        features = self.prepare_features(data)

        # Get probability
        probability = self.model.predict_proba(features)[0, 1]

        # Make prediction using optimal threshold
        prediction = (
            "All-Star" if probability >= self.optimal_threshold else "Non-All-Star"
        )

        # Determine confidence level
        if probability >= 0.75 or probability <= 0.25:
            confidence = "High"
        elif probability >= 0.60 or probability <= 0.40:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Get feature importance for this prediction
        top_positive, top_negative = self._get_feature_importance(features)

        return {
            "prediction": prediction,
            "probability": float(probability),
            "confidence": confidence,
            "threshold_used": self.optimal_threshold,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }

    def _get_feature_importance(
        self, features: np.ndarray, top_n: int = 5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get top features contributing to this prediction.

        Args:
            features: Feature array for the prediction
            top_n: Number of top features to return

        Returns:
            Tuple of (positive_features, negative_features) dicts
        """
        # Get model coefficients
        coefficients = self.model.coef_[0]

        # Get feature values (convert sparse to dense)
        feature_values = features.toarray()[0]

        # Calculate contribution (coefficient * feature_value)
        contributions = coefficients * feature_values

        # Get indices of non-zero contributions
        nonzero_indices = np.where(feature_values != 0)[0]

        if len(nonzero_indices) == 0:
            return {}, {}

        # Get top positive and negative contributions
        nonzero_contributions = contributions[nonzero_indices]
        nonzero_feature_names = [
            self.feature_names[i] if self.feature_names else f"feature_{i}"
            for i in nonzero_indices
        ]

        # Sort by contribution
        sorted_indices = np.argsort(nonzero_contributions)

        # Top negative (predicting Non-All-Star)
        top_negative_idx = sorted_indices[:top_n]
        top_negative = {
            nonzero_feature_names[i]: float(nonzero_contributions[i])
            for i in top_negative_idx
        }

        # Top positive (predicting All-Star)
        top_positive_idx = sorted_indices[-top_n:][::-1]
        top_positive = {
            nonzero_feature_names[i]: float(nonzero_contributions[i])
            for i in top_positive_idx
        }

        return top_positive, top_negative

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": type(self.model).__name__ if self.model else None,
            "optimal_threshold": self.optimal_threshold,
            "num_features": len(self.feature_names) if self.feature_names else None,
            "numerical_features": self.numerical_features,
        }
