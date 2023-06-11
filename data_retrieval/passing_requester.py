import requests
import pandas as pd
import urllib.parse
from constants import HEADERS

PER_MODES = ['Totals','PerGame','Per100Possessions','Per100Plays','Per48',
    'Per40','Per36','Per1','PerPossession','PerPlay','MinutesPer']

class PlayerPassingRequester:

    def __init__(self):
        self.rows = []
        self.player_info_url = 'https://stats.nba.com/stats/playerdashptpass'


    def populate_player(self, season_id, per_mode='PerGame', date_from='', date_to='', player_id='1628378'):
        params = self.build_params(season_id, per_mode, date_from, date_to, player_id)
        # Encode without safe '+', apparently the NBA likes unsafe url params.
        params_str = urllib.parse.urlencode(params, safe=':+')
        response = requests.get(url=self.player_info_url, headers=HEADERS, params=params_str).json()
        df = pd.json_normalize(response, 'resultSets')
        # first is passes made, second is passes received
        df = pd.DataFrame(df['rowSet'][0], columns = df['headers'][0])
        self.df = df
        # return df

    def build_params(self, season_id, per_mode='PerGame', date_from='', date_to='', player_id='1628378'):
        params = {
            'DateFrom': date_from,
            'DateTo': date_to,
            'GameSegment': '',
            'LastNGames': '0',
            'LeagueID': '00',
            'Location': '',
            'Month': '0',
            'OpponentTeamID': '0',
            'Outcome': '',
            'PORound': '0',
            'PerMode': per_mode,
            'Period': '0',
            'PlayerID': player_id,
            'Season': season_id,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'TeamID': '0',
            'VsConference': '',
            'VsDivision': '',
        }
        return params