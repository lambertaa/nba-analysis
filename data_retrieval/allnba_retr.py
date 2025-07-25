import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

URL = "https://www.nba.com/news/history-all-nba-teams"


def allnba_retr() -> pd.DataFrame:
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    season = None

    records = []
    season = None
    nteam = None
    nteam_map = {"FIRST TEAM": 1, "SECOND TEAM": 2, "THIRD TEAM": 3}

    # The article's div containing all data
    article_div = soup.find("div", class_="ArticleContent_article__NBhQ8")

    for tag in article_div.find_all(["p", "span", "strong"]):
        raw_text = tag.get_text(strip=True)
        text = raw_text.upper()

        # Detect new season (e.g., > 2023-24)
        season_match = re.search(r"\d{4}-\d{2}", text)
        if season_match:
            season = season_match.group(0)
            continue

        # Detect team designation
        for key in nteam_map:
            if key in text:
                nteam = nteam_map[key]
                break

        # Parse player lines
        if "•" in raw_text and "," in raw_text and season and nteam:
            lines = raw_text.split("•")
            for line in lines:
                line = line.strip()
                if not line or "," not in line:
                    continue
                parts = line.split(",", 1)
                player = parts[0].strip("• ").strip()
                team = parts[1].strip()
                records.append(
                    {"PLAYER": player, "TEAM": team, "SEASON": season, "NTEAM": nteam}
                )

    # Convert to DataFrame and ensure PLAYER is not the index
    df = pd.DataFrame(records)
    df.reset_index(drop=True, inplace=True)
    return df
