"""
Quick test of the API without running a server
Tests the predictor directly
"""

from predictor import NBADraftPredictor
import json


def test_predictor():
    """Test the predictor directly"""
    print("=" * 60)
    print("Testing NBA Draft Predictor")
    print("=" * 60 + "\n")

    # Initialize predictor
    print("Loading predictor...")
    predictor = NBADraftPredictor()
    print("✓ Predictor loaded successfully!\n")

    # Test 1: Elite prospect
    print("Test 1: Elite Prospect")
    print("-" * 40)
    data1 = {
        "Strengths": "Elite three point shooter with deep range. High basketball IQ and excellent court vision. Strong ball handling ability and can create his own shot. Good defensive awareness and quick hands. NBA ready body and mature game.",
        "Weaknesses": "Average lateral quickness on defense. Needs to improve finishing at the rim through contact. Can be turnover prone when pressured. Limited rebounding for his size.",
        "overall": 88.0,
        "Athleticism": 75.0,
        "Size": 80.0,
        "Defense": 70.0,
        "Strength": 70.0,
        "Quickness": 80.0,
        "Leadership": 85.0,
        "JumpShot": 92.0,
        "NBAReady": 85.0,
    }

    result1 = predictor.predict(data1)
    print(json.dumps(result1, indent=2))
    print()

    # Test 2: Raw athletic prospect
    print("Test 2: Raw Athletic Prospect")
    print("-" * 40)
    data2 = {
        "Strengths": "Tremendous leaping ability and elite athleticism. Great physical tools with long wingspan. High energy player who runs the floor well. Defensive potential with quick first step.",
        "Weaknesses": "Very limited offensive game. Poor shooter with inconsistent mechanics. Needs to add weight and strength. Low basketball IQ and poor decision making. Turnover prone.",
        "overall": 72.0,
        "Athleticism": 95.0,
        "Size": 85.0,
        "Defense": 78.0,
        "Strength": 65.0,
        "Quickness": 90.0,
        "Leadership": 55.0,
        "JumpShot": 45.0,
        "NBAReady": 50.0,
    }

    result2 = predictor.predict(data2)
    print(json.dumps(result2, indent=2))
    print()

    # Test 3: Polished but limited upside
    print("Test 3: Polished Prospect")
    print("-" * 40)
    data3 = {
        "Strengths": "NBA ready body and game. Excellent fundamentals and high basketball IQ. Consistent mid range jumper. Good team defender. Mature player who makes smart decisions.",
        "Weaknesses": "Limited upside and ceiling. Average athleticism. Lacks elite physical tools. Not a great three point shooter. Limited ability to create own shot.",
        "overall": 78.0,
        "Athleticism": 65.0,
        "Size": 75.0,
        "Defense": 80.0,
        "Strength": 75.0,
        "Quickness": 65.0,
        "Leadership": 85.0,
        "JumpShot": 75.0,
        "NBAReady": 90.0,
    }

    result3 = predictor.predict(data3)
    print(json.dumps(result3, indent=2))
    print()

    # Model info
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    info = predictor.get_model_info()
    print(json.dumps(info, indent=2))
    print()

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_predictor()
