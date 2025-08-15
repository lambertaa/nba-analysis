from io import StringIO
from bs4 import BeautifulSoup
import pandas as pd
import requests


def get_missed_games(year):
    url_format = "https://www.spotrac.com/nba/injured/_/year/{year}/view/player"

    # Send a GET request to the page URL
    response = requests.get(url_format.format(year=year))

    if response.status_code != 200:
        print(
            f"Failed to retrieve data for {year}. Status code: {response.status_code}"
        )

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", class_="table", id="table")
    if table is None:
        print(f"No table found for {year}.")
        return None
    df = pd.read_html(StringIO(str(table)))[0]
    return df
