import requests
import pandas as pd
import urllib.parse
from constants import HEADERS

PER_MODES = [
    "Totals",
    "PerGame",
    "Per100Possessions",
    "Per100Plays",
    "Per48",
    "Per40",
    "Per36",
    "Per1",
    "PerPossession",
    "PerPlay",
    "MinutesPer",
]


class PlayerSeasonRequester:
    def __init__(self):
        self.df = pd.DataFrame()
        self.player_info_url = "https://stats.nba.com/stats/leaguedashplayerstats"

    def populate_season(self, season_id, per_mode="PerGame", **kwargs):
        params = self.build_params(season_id, per_mode, **kwargs)
        response = requests.get(
            url=self.player_info_url,
            headers=HEADERS,
            params=params,
            timeout=10,  # timeout prevents hanging
        )
        response.raise_for_status()  # good practice to catch HTTP errors
        json_data = response.json()
        df = pd.DataFrame(
            json_data["resultSets"][0]["rowSet"],
            columns=json_data["resultSets"][0]["headers"],
        )
        self.df = df
        return df

    def build_params(
        self, season_id, per_mode="PerGame", PlayerExperience="", MeasureType="Base"
    ):
        return {
            "College": "",
            "Conference": "",
            "Country": "",
            "DateFrom": "",
            "DateTo": "",
            "Division": "",
            "DraftPick": "",
            "DraftYear": "",
            "GameScope": "",
            "GameSegment": "",
            "Height": "",
            "ISTRound": "",  # Include this; it's new for in-season tournament
            "LastNGames": "0",
            "LeagueID": "00",
            "Location": "",
            "MeasureType": MeasureType,
            "Month": "0",
            "OpponentTeamID": "0",
            "Outcome": "",
            "PORound": "0",
            "PaceAdjust": "N",
            "PerMode": per_mode,
            "Period": "0",
            "PlayerExperience": PlayerExperience,
            "PlayerPosition": "",
            "PlusMinus": "N",
            "Rank": "N",
            "Season": season_id,
            "SeasonSegment": "",
            "SeasonType": "Regular Season",  # or "Regular Season"
            "ShotClockRange": "",
            "StarterBench": "",
            "TeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "Weight": "",
        }


# class PlayerSeasonRequesterTotal(PlayerSeasonRequester):

#     per_mode = 'Totals'
