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
        self.rows = []
        self.player_info_url = "http://stats.nba.com/stats/leaguedashplayerstats"

    def populate_season(self, season_id, per_mode="PerGame", **kwargs):
        params = self.build_params(season_id, per_mode, **kwargs)
        # Encode without safe '+', apparently the NBA likes unsafe url params.
        params_str = urllib.parse.urlencode(params, safe=":+")
        response = requests.get(
            url=self.player_info_url, headers=HEADERS, params=params_str
        ).json()
        df = pd.json_normalize(response, "resultSets")
        df = pd.DataFrame(df["rowSet"][0], columns=df["headers"][0])
        self.df = df
        # return df

    def build_params(
        self, season_id, per_mode="PerGame", PlayerExperience="", MeasureType="base"
    ):
        params = (
            ("College", ""),
            ("Conference", ""),
            ("Country", ""),
            ("DateFrom", ""),
            ("DateTo", ""),
            ("Division", ""),
            ("DraftPick", ""),
            ("DraftYear", ""),
            ("GameScope", ""),
            ("GameSegment", ""),
            ("Height", ""),
            ("LastNGames", "0"),
            ("LeagueID", "00"),
            ("Location", ""),
            ("MeasureType", MeasureType),
            ("Month", "0"),
            ("OpponentTeamID", "0"),
            ("Outcome", ""),
            ("PORound", "0"),
            ("PaceAdjust", "N"),
            ("PerMode", per_mode),
            ("Period", "0"),
            ("PlayerExperience", PlayerExperience),
            ("PlayerPosition", ""),
            ("PlusMinus", "N"),
            ("Rank", "N"),
            ("Season", season_id),
            ("SeasonSegment", ""),
            ("SeasonType", "Regular Season"),
            ("ShotClockRange", ""),
            ("StarterBench", ""),
            ("TeamID", "0"),
            ("TwoWay", "0"),
            ("VsConference", ""),
            ("VsDivision", ""),
            ("Weight", ""),
        )
        return params


# class PlayerSeasonRequesterTotal(PlayerSeasonRequester):

#     per_mode = 'Totals'
