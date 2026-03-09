"""
Example API Requests

Test the NBA Draft Prediction API with example requests.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_predict_by_name():
    """Test prediction by player name"""
    print("Testing prediction by player name...")

    # Example: Cooper Flagg (2025 draft prospect)
    payload = {"player_name": "Cooper Flagg"}

    response = requests.post(f"{BASE_URL}/predict/by-name", json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")


def test_predict_by_report():
    """Test prediction by scouting report"""
    print("Testing prediction by scouting report...")

    # Example scouting report
    payload = {
        "strengths": "Elite three point shooter with deep range. High basketball IQ and excellent court vision. Strong ball handling ability and can create his own shot. Good defensive awareness and quick hands. NBA ready body and mature game.",
        "weaknesses": "Average lateral quickness on defense. Needs to improve finishing at the rim through contact. Can be turnover prone when pressured. Limited rebounding for his size.",
        "overall": 88.0,
        "athleticism": 75.0,
        "size": 80.0,
        "defense": 70.0,
        "rebounding": 65.0,
        "jump_shot": 92.0,
        "nba_ready": 85.0,
    }

    response = requests.post(f"{BASE_URL}/predict/by-report", json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")


def test_predict_raw_prospect():
    """Test prediction for a raw, high-upside prospect"""
    print("Testing prediction for raw prospect...")

    payload = {
        "strengths": "Tremendous leaping ability and elite athleticism. Great physical tools with long wingspan. High energy player who runs the floor well. Defensive potential with quick first step.",
        "weaknesses": "Very limited offensive game. Poor shooter with inconsistent mechanics. Needs to add weight and strength. Low basketball IQ and poor decision making. Turnover prone.",
        "overall": 72.0,
        "athleticism": 95.0,
        "size": 85.0,
        "defense": 78.0,
        "rebounding": 70.0,
        "jump_shot": 45.0,
        "nba_ready": 50.0,
    }

    response = requests.post(f"{BASE_URL}/predict/by-report", json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")


def test_predict_polished_prospect():
    """Test prediction for a polished, NBA-ready prospect"""
    print("Testing prediction for polished prospect...")

    payload = {
        "strengths": "NBA ready body and game. Excellent fundamentals and high basketball IQ. Consistent mid range jumper. Good team defender. Mature player who makes smart decisions.",
        "weaknesses": "Limited upside and ceiling. Average athleticism. Lacks elite physical tools. Not a great three point shooter. Limited ability to create own shot.",
        "overall": 78.0,
        "athleticism": 65.0,
        "size": 75.0,
        "defense": 80.0,
        "rebounding": 72.0,
        "jump_shot": 75.0,
        "nba_ready": 90.0,
    }

    response = requests.post(f"{BASE_URL}/predict/by-report", json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("NBA Draft Prediction API - Example Requests")
    print("=" * 60 + "\n")

    # Run tests
    test_health_check()
    test_predict_by_name()  # Test web scraping
    test_predict_by_report()
    test_predict_raw_prospect()
    test_predict_polished_prospect()

    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
