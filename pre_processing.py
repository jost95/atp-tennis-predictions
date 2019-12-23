import pandas as pd
import numpy as np

def pre_process():
    # Read processed data
    mutual_matches = pd.read_csv('input/fixed/mutual_games_to_20181231.csv')
    mutual_matches_clay = pd.read_csv('input/fixed/mutual_matches_clay_to_20181231.csv.csv')
    mutual_matches_grass = pd.read_csv('input/fixed/mutual_matches_grass_to_20181231.csv.csv')
    mutual_matches_hard = pd.read_csv('input/fixed/mutual_matches_hard_to_20181231.csv.csv')
    mutual_games = pd.read_csv('input/fixed/mutual_games_to_20181231.csv')

