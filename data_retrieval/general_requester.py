import requests
from constants import HEADERS
import pandas as pd

class GenericRequester:

    def __init__(self, settings, url, table):
        self.settings = settings
        self.url = url
        self.table = table

    def generate_df(self, params):
        """
        Request from API and build dataframe
        """
        response = requests.get(url=self.url, headers=HEADERS, params=params).json()
        df = pd.json_normalize(response, 'resultSets')
        df = pd.DataFrame(df['rowSet'][0], columns=df['headers'][0])
        self.df = df
    
    def populate_db(self):
        """
        Bulk populate db table
        """
        print('placeholder')