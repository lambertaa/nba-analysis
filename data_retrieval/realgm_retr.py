import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Union
from io import StringIO


def get_realgm_allstar_roster(year: int) -> Union[pd.DataFrame, None]:
    base_url = "https://basketball.realgm.com/nba/allstar/game/rosters"

    years = list(range(1951, 2025))

    # Send a GET request to the page URL
    response = requests.get(f"{base_url}/{year}")

    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table", class_="table-compact")

        year_df_list = []
        for table in tables:
            year_df_list.append(pd.read_html(StringIO(str(table)))[0])

        df = pd.concat(year_df_list).reset_index(drop=True)
        # remove duplicate rows based on Player
        df = df.drop_duplicates(subset=["Player"])
        return df
    else:
        print(
            f"Failed to retrieve data for {year}. Status code: {response.status_code}"
        )
        return None


def get_realgm_allstar_rosters(years: list[int]) -> Union[pd.DataFrame, None]:
    allstar_rosters = []
    for year in years:
        roster = get_realgm_allstar_roster(year)
        if roster is not None:
            roster["Year"] = year
            allstar_rosters.append(roster)

    # Concatenate all the dataframes into a single dataframe
    allstar_rosters_df = pd.concat(allstar_rosters, ignore_index=True)
    return (
        allstar_rosters_df.reset_index(drop=True)
        if not allstar_rosters_df.empty
        else None
    )
