# This script extract all players from 2019 and saves them to a new csv file

import numpy as np
import pandas as pd

# Read data
matches_2019 = pd.read_csv('../input/raw/atp_matches_2019.csv')
all_players = pd.read_csv('../input/raw/atp_players.csv')

# Extract players
winner_ids = matches_2019['winner_id'].to_numpy()
loser_ids = matches_2019['loser_id'].to_numpy()

# Sort out unique players
player_ids = np.unique(np.append(winner_ids, loser_ids))

# Extract player info, column names needs to be added for below to work
players_2019 = all_players.loc[all_players['id'].isin(player_ids)]

# Save to new csv file
players_2019.to_csv('../input/fixed/atp_players_2019.csv', index=False, float_format='%.f')
