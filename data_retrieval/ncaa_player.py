import pandas as pd

url_format = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"

column_headers = [
    "player_name",
    "team",
    "conf",
    "GP",
    "Min_per",
    "ORtg",
    "usg",
    "eFG",
    "TS_per",
    "ORB_per",
    "DRB_per",
    "AST_per",
    "TO_per",
    "FTM",
    "FTA",
    "FT_per",
    "twoPM",
    "twoPA",
    "twoP_per",
    "TPM",
    "TPA",
    "TP_per",
    "blk_per",
    "stl_per",
    "ftr",
    "yr",
    "ht",
    "num",
    "porpag",
    "adjoe",
    "pfr",
    "year",
    "pid",
    "type",
    "Rec Rank",
    "ast/tov",
    "rimmade",
    "rimmade+ri",
    "midmade",
    "midmade+m",
    "rimmade/(ri",
    "midmade/(m",
    "dunksmade",
    "dunksmiss+",
    "dunksmade/",
    "pick",
    "drtg",
    "adrtg",
    "dporpag",
    "stops",
    "bpm",
    "obpm",
    "dbpm",
    "gbpm",
    "mp",
    "ogbpm",
    "dgbpm",
    "oreb",
    "dreb",
    "treb",
    "ast",
    "stl",
    "blk",
    "pts",
    "role",
    "3p/100?",
]


def get_ncaa_player_data(year: int) -> pd.DataFrame:
    """Fetch NCAA player data for a given year from Bart Torvik's website.
    Args:
        year (int): The year for which to fetch the player data.
    Returns:
        pd.DataFrame: A DataFrame containing the player data for the specified year.
    """

    url = url_format.format(year=year)
    df = pd.read_csv(url, header=None, names=column_headers)

    if df.empty:
        raise ValueError(
            f"No data found for year {year}. Please check the year and try again."
        )

    return df


def get_ncaa_player_data_for_years(years: list[int]) -> pd.DataFrame:
    """Fetch NCAA player data for multiple years.
    Args:
        years (list[int]): A list of years for which to fetch the player data.
    Returns:
        pd.DataFrame: A DataFrame containing the player data for the specified years.
    """
    all_data = []
    for year in years:
        try:
            df = get_ncaa_player_data(year)
            df["year"] = year  # Add a column for the year
            all_data.append(df)
        except ValueError as e:
            print(e)

    if not all_data:
        raise ValueError("No data was retrieved for the specified years.")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
