import pandas as pd

URL_FORMAT = 'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
DB_URI = 'D:/Andy/python/nba_analysis_app/nba_stats.sqlite'

def per_game_request(year):
    import requests
    from bs4 import BeautifulSoup as BS

    url = URL_FORMAT.format(year=year)
    r = requests.get(url)
    r_html = r.text

    soup = BS(r_html,'html.parser')

    table = soup.find_all(class_='full_table')
    head = soup.find(class_='thead')
    colnames = []
    for item in head:
        s = item.text.strip()
        if s != '':
            colnames.append(s)

    players = []
    for i in range(len(table)):
        player_ = []
        for td in table[i].find_all('td'):
            player_.append(td.text)
        players.append(player_)

    df = pd.DataFrame(players, columns=colnames[1:]).set_index('Player')
    numeric_cols = [col for col in list(df.columns) if col not in ['Pos','Tm']]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    return df

def per_game2sqlite(year):
    import sqlite3 as sql
    df = per_game_request(year)
    df['SeasonYearEnd'] = year

    con = sql.connect(DB_URI)
    df.to_sql(name='per_game', con=con, if_exists='replace')

    # https://github.com/mpope9/nba-sql