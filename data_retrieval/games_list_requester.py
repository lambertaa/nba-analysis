import os
import sys

# sys.path.append(os.path.abspath("."))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import urllib.parse

# from constants import HEADERS

HEADERS = {
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": (
        #'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) '
        #'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130'
        #'Safari/537.36'
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Referer": "https://stats.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}


class GamesListRequester:
    def __init__(self):
        self.rows = []
        self.games_list_url = "https://stats.nba.com/stats/leaguegamelog"
        # self.player_info_url = "http://stats.nba.com/stats/leaguedashplayerstats"

    def get_games_list(self, season_id, season_type="Regular Season", **kwargs):
        params = self.build_params(season_id, season_type, **kwargs)
        # Encode without safe '+', apparently the NBA likes unsafe url params.
        params_str = urllib.parse.urlencode(params, safe=":+")
        response = requests.get(
            url=self.games_list_url, headers=HEADERS, params=params_str
        ).json()
        result_set = response["resultSets"][
            0
        ]  # there's usually only one in this endpoint
        rows = result_set["rowSet"]
        columns = result_set["headers"]

        df = pd.DataFrame(rows, columns=columns)
        self.df = df

    def build_params(self, season_id, season_type="Regular Season", **kwargs):
        params = (
            ("LeagueID", "00"),
            ("Season", season_id),
            ("SeasonType", season_type),
            ("PlayerOrTeam", "T"),
            ("Counter", 1000),
            ("Sorter", "DATE"),
            ("Direction", "DESC"),
            ("DateFrom", ""),
            ("DateTo", ""),
        )

        return params
