import numpy as np
from numpy.random import choice
from PIL import ImageTk, Image
import os

file_dir = os.path.dirname(os.path.realpath(__file__))


class Lottery:
    def __init__(self):
        self._team_rank = {
            "Yo Yo Ma": 3,
            "Honey Badger": 9,
            "Smitty Werbenjagermanjensen": 10,
            "Steel Magnolias": 4,
            "Big Flair Day": 11,
            "Gwaggies": 2,
            "BUZZSAW": 6,
            "KingKong": 7,
            "(IR)monstar": 12,
            "Dumpster Fire": 1,
            "Lupita and Los Santos": 8,
            "Mitchell Don Got Me Jaded": 5,
        }
        self._team_rank = {
            k: v for k, v in sorted(self._team_rank.items(), key=lambda item: item[1])
        }
        self._odds_dict = {
            12: 15,
            11: 15,
            10: 15,
            9: 13,
            8: 10.5,
            7: 9,
            6: 7.5,
            5: 6,
            4: 4.5,
            3: 3,
            2: 1,
            1: 0.5,
        }
        self._team_rank_inverse = {v: k for k, v in self._team_rank.items()}
        self._lotto_balls = {
            rank: int(odds / 100 * 1000) for rank, odds in self._odds_dict.items()
        }
        self.pick_order = []
        self.set_team_images()
        self.n_stats_picks_max = 3
        self.n_stats_picks = 0
        self.stats_teams = []

    def teams(self):
        return list(self._team_rank.keys())

    def get_team_rank(self):
        return self._team_rank

    def get_odds(self):
        return self._odds_dict

    def update_odds(self):
        N = float(sum(self._lotto_balls.values()))
        try:
            self._odds_dict = {
                rank: balls / N * 100 for rank, balls in self._lotto_balls.items()
            }
        except ZeroDivisionError:
            team_name = self.stats_teams.pop(0)
            team_rank = self._team_rank[team_name]
            for rank, odds in self._odds_dict.items():
                if rank == team_rank:
                    self._lotto_balls[rank] = 1
                else:
                    self._lotto_balls[rank] = 0
            self.update_odds()

    def get_pick(self):
        N = float(sum(self._lotto_balls.values()))
        p = [self._lotto_balls[c] / N for c in self._lotto_balls]
        draw = choice(list(self._lotto_balls.keys()), 1, p=p)[0]
        self.pick_order.append(self._team_rank_inverse[draw])
        del self._lotto_balls[draw]
        self.update_odds()

    def set_team_images(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.team_images = {}
        for team in self._team_rank.keys():
            image_path = os.path.join(file_dir, f"profile_pics/{team}.jpg")
            image = Image.open(image_path)
            resized_image = image.resize((200, 200), Image.ANTIALIAS)
            self.team_images[team] = resized_image

    def set_odds_to_zero(self, team_name):
        # Find the team rank from the team name
        team_rank = self._team_rank[team_name]
        # Set the lotto balls for this team rank to zero
        self._lotto_balls[team_rank] = 0
        # Update odds to reflect the change
        self.update_odds()


lottery = Lottery()


def get_random_team():
    import time

    global lottery, winner_lbl

    nsec = 1
    # nsec = choice(list(range(3, 11)))
    start = time.time()
    while (time.time() - start) < nsec:
        # for i in range(1000):
        N = float(sum(lottery._lotto_balls.values()))
        p = [lottery._lotto_balls[c] / N for c in lottery._lotto_balls]
        draw = choice(list(lottery._lotto_balls.keys()), 1, p=p)[0]
        team_name = lottery._team_rank_inverse[draw]
        winner_lbl.config(text=team_name)
        display_image(team_name)
        tkWindow.update()
        time.sleep(0.1)


def get_and_display_pick():
    global lottery, winner_lbl, tkWindow
    lottery.get_pick()
    winner_lbl.config(text=lottery.pick_order[-1])
    tkWindow.update()


def update_team_labels():
    global tkWindow, lottery
    teams_copy = lottery._team_rank_inverse.copy()
    for rank, odds in lottery.get_odds().items():
        name = teams_copy.pop(rank)
        # name = lottery._team_rank_inverse[rank]
        label = tkWindow.nametowidget(f"team{rank}")
        print_odds = "{:.1f}".format(odds)
        label.config(text=f"{name}: {print_odds}%")
        tkWindow.update()

    # delete remaining labels
    for rank, name in teams_copy.items():
        label = tkWindow.nametowidget(f"team{rank}")
        label.grid_remove()


def get_pick_clicked():
    from tkinter import DISABLED
    from tkinter import messagebox

    global lottery
    global button
    from tkinter import NORMAL

    global button, lottery
    get_random_team()
    get_and_display_pick()

    selected_team = lottery.pick_order[-1]

    button.config(
        text="Congratulations!\nSelect your draft slot!", state=DISABLED, cursor=""
    )

    # update_team_labels()
    display_image(selected_team)

    # Ask user if they want to keep the slot or choose an option to set odds to zero
    if lottery.n_stats_picks < lottery.n_stats_picks_max:
        response = messagebox.askyesno(
            "Choose an Option",
            f"Would you like to take the helper stats instead of picking a draft slot?",
        )

        if response:  # User selected "Yes"
            lottery.set_odds_to_zero(selected_team)
            button.config(text="Get pick", state=NORMAL, cursor="hand2")
            update_team_labels()
            lottery.n_stats_picks += 1
            lottery.stats_teams.append(selected_team)
            return

    update_team_labels()


def update_label(e):
    from tkinter import NORMAL

    global lottery, button
    e.widget.config(text=lottery.pick_order[-1], cursor="")
    e.widget.unbind("<Button-1>")
    button.config(text="Get pick", state=NORMAL, cursor="hand2")


def display_image(team_name):
    global image_label, lottery

    # Create a label widget to display the resized image
    if "image_label" in globals():
        photo = ImageTk.PhotoImage(lottery.team_images[team_name])
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to the image to prevent it from being garbage collected
    else:
        image = Image.open(
            "D:/Andy/python/nba-analysis/nba_fantasy/profile_pics/please_wait.jpg"
        )
        resized_image = image.resize((200, 200), Image.ANTIALIAS)

        # Convert the resized image to a format that tkinter can display
        photo = ImageTk.PhotoImage(resized_image)
        image_label = tk.Label(tkWindow, image=photo)
        image_label.image = photo

        image_label.grid(
            row=7, column=1, rowspan=10, columnspan=1
        )  # Adjust row and column as needed


import tkinter as tk

tkWindow = tk.Tk()
tkWindow.geometry("1100x600+700+350")
tkWindow.title("2022-2023 NBA Fantasy Draft Lottery")
button = button_get_pick = tk.Button(
    tkWindow,
    text="Get\nPick",
    command=get_pick_clicked,
    cursor="hand2",
    font=("Arial", 20),
    width=20,
)
button.grid(row=3, column=1, rowspan=2)
winner_lbl = tk.Label(
    tkWindow,
    text="Welcome!",
    font=("Arial", 25),
    padx=2.5,
    pady=20,
    width=30,
    bg="#d9d8d7",
)
winner_lbl.grid(row=5, column=1, rowspan=2)
winner_labels = []
for i, name in enumerate(reversed(lottery.teams())):
    winner_label = tk.Label(
        tkWindow,
        text=f"Pick {i+1}",
        font=21,
        padx=10,
        pady=5,
        bg="#d9d8d7",
        width=30,
        cursor="hand2",
    )
    winner_label.bind("<Button-1>", lambda e: update_label(e))
    winner_label.grid(row=i, column=2, pady=5, padx=5)
    rank = lottery.get_team_rank()[name]
    odds = lottery.get_odds()[rank]
    team_label = tk.Label(
        tkWindow, text=f"{name}: {odds}%", name=f"team{rank}", width=30
    )
    team_label.grid(row=i, column=0)

display_image(team_name=None)
tkWindow.update()
tkWindow.mainloop()
