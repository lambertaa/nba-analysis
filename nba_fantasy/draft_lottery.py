import numpy as np
from numpy.random import choice
from PIL import ImageTk, Image
import os

class Lottery():
    def __init__(self):

        self._team_rank= {
            'Yo Yo Ma': 1,
            'Honey Badger': 2,
            'Smitty Werbenjagermanjensen': 3,
            'Steel Magnolias': 4,
            'Big Hair Day': 5,
            'Gwaggies': 6,
            'BUZZSAW': 7,
            'King Kong': 8,
            '(IR)monstar': 9,
            'Dumpster Fire': 10,
            'Lupita and Los Santos': 11,
            'Mitchell Don Got Me Psyched': 12
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
            1: 0.5
        }
        self._team_rank_inverse = {v:k for k,v in self._team_rank.items()}
        self._lotto_balls = {rank: int(odds / 100 * 1000) for rank, odds in self._odds_dict.items()}
        self.pick_order = []

    def teams(self):
        return list(self._team_rank.keys())

    def get_team_rank(self):
        return self._team_rank

    def get_odds(self):
        return self._odds_dict

    def update_odds(self):
        N = float(sum(self._lotto_balls.values()))
        self._odds_dict = {rank: balls / N * 100 for rank, balls in self._lotto_balls.items()}

    def get_pick(self):
        N = float(sum(self._lotto_balls.values()))
        p = [self._lotto_balls[c] / N for c in self._lotto_balls]
        print(p)
        draw = choice(list(self._lotto_balls.keys()), 1, p=p)[0]
        self.pick_order.append(self._team_rank_inverse[draw])
        del self._lotto_balls[draw]
        self.update_odds()

lottery = Lottery()

def get_random_team():
    import time
    global lottery, winner_lbl

    nsec = choice(list(range(3,11)))
    start = time.time()
    while (time.time() - start) < nsec:
    # for i in range(1000):
        N = float(sum(lottery._lotto_balls.values()))
        p = [lottery._lotto_balls[c] / N for c in lottery._lotto_balls]
        draw = choice(list(lottery._lotto_balls.keys()), 1, p=p)[0]
        team_name = lottery._team_rank_inverse[draw]
        winner_lbl.config(text=team_name)
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
        label = tkWindow.nametowidget(f'team{rank}')
        print_odds = '{:.1f}'.format(odds)
        label.config(text=f'{name}: {print_odds}%')
        tkWindow.update()

    # delete remaining labels
    for rank, name in teams_copy.items():
        label = tkWindow.nametowidget(f'team{rank}')
        label.grid_remove()

def update_image():
    global image_label, lottery, file_dir
    image = Image.open(os.path.join(file_dir, 'profile_pics\Mithcell Don Got Me Psyched.jpg'))
    # image = Image.open(os.path.join(file_dir,f'profile_pics\{lottery.pick_order[-1]}.jpg'))
    image = image.resize((200,200))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)

def get_pick_clicked():
    from tkinter import DISABLED
    global button, lottery
    get_random_team()
    get_and_display_pick()
    button.config(text='Congratulations!\nSelect your draft slot!',
        state=DISABLED, cursor='')
    update_team_labels()
    # update_image()

def update_label(e):
    from tkinter import NORMAL
    global lottery, button
    e.widget.config(text=lottery.pick_order[-1], cursor='')
    e.widget.unbind('<Button-1>')
    button.config(text='Get pick', state=NORMAL, cursor='hand2')

import tkinter as tk
tkWindow = tk.Tk()
tkWindow.geometry('1100x600+700+350')
tkWindow.title('2022-2023 NBA Fantasy Draft Lottery')
button = button_get_pick = tk.Button(tkWindow,
    text='Get\nPick',
    command=get_pick_clicked,
    cursor='hand2',
    font=("Arial", 20),
    width=20)
button.grid(row=3,column=1, rowspan=2)
winner_lbl = tk.Label(tkWindow,
    text='Winner',
    font=("Arial", 25),
    padx=2.5, 
    pady=20,
    width=30,
    bg='#d9d8d7')
winner_lbl.grid(row=5, column=1, rowspan=2)
# file_dir = os.path.dirname(os.path.realpath(__file__))
# image = Image.open(os.path.join(file_dir,'profile_pics\Hayward.jpg'))
# image = image.resize((200,200))
# image = image.thumbnail((30,30))
# photo = ImageTk.PhotoImage(image)
# image_label = tk.Label(tkWindow, image=photo)
# image_label.grid(row=3, column=1, rowspan=4)
winner_labels = []
for i, name in enumerate(reversed(lottery.teams())):
    winner_label = tk.Label(tkWindow, text=f'Pick {i+1}',
        font=21, padx=10, pady=5, bg='#d9d8d7', width=30,
        cursor='hand2')
    winner_label.bind('<Button-1>', lambda e:update_label(e))
    winner_label.grid(row=i,column=2, pady=5, padx=5)
    rank = lottery.get_team_rank()[name]
    odds = lottery.get_odds()[rank]
    team_label = tk.Label(tkWindow, text=f'{name}: {odds}%', name=f'team{rank}',
        width=30)
    team_label.grid(row=i, column=0)

# tkWindow.grid_columnconfigure(1, weight=1)
# tkWindow.grid_rowconfigure(3, weight=1)
tkWindow.update()
tkWindow.mainloop()