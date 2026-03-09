"""
Quick test for player name prediction
Requires the API server to be running
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_player(player_name):
    """Test prediction for a specific player"""
    print(f"\n{'='*60}")
    print(f"Testing prediction for: {player_name}")
    print('='*60)
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=2)
        if health.status_code != 200:
            print("❌ API server is not healthy!")
            return
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running!")
        print("\nPlease start the server first:")
        print("  python -m uvicorn app:app --host 0.0.0.0 --port 8000")
        return
    
    # Make prediction request
    payload = {"player_name": player_name}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/by-name",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!\n")
            print(f"Player: {result.get('player_name', player_name)}")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability: {result['probability']:.1%}")
            print(f"Confidence: {result['confidence']}")
            print(f"\nTop Positive Features:")
            for feature, value in list(result['top_positive_features'].items())[:3]:
                print(f"  • {feature}: {value:.3f}")
            print(f"\nTop Negative Features:")
            for feature, value in list(result['top_negative_features'].items())[:3]:
                print(f"  • {feature}: {value:.3f}")
        else:
            print(f"\n❌ Error {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (scraping may have failed)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Get player name from command line or use default
    if len(sys.argv) > 1:
        player_name = " ".join(sys.argv[1:])
    else:
        player_name = "Cooper Flagg"  # Default 2025 prospect
    
    test_player(player_name)
    
    print("\n" + "="*60)
    print("To test other players, run:")
    print('  python test_player_name.py "Player Name"')
    print("="*60)

