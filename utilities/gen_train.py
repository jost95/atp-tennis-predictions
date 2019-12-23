import pandas as pd
import numpy as np


def gen_train(base_weight, from_year, to_year):
    # Read processed data
    mutual_matches = pd.read_csv('../input/fixed/mutual_matches_to_' + str(to_year - 1) + '.csv')
    mutual_matches_clay = pd.read_csv('../input/fixed/mutual_matches_clay_to_' + str(to_year - 1) + '.csv')
    mutual_matches_grass = pd.read_csv('../input/fixed/mutual_matches_grass_to_' + str(to_year - 1) + '.csv')
    mutual_matches_hard = pd.read_csv('../input/fixed/mutual_matches_hard_to_' + str(to_year - 1) + '.csv')
    mutual_games = pd.read_csv('../input/fixed/mutual_games_to_' + str(to_year - 1) + '.csv')
    cond_stats = pd.read_csv('../input/fixed/cond_stats_to_' + str(to_year - 1) + '.csv')

    # Read all matches
    matches = []

    for year in range(from_year, to_year + 1):
        matches.append(
            pd.read_csv('../input/raw/atp_matches_futures_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv('../input/raw/atp_matches_qual_chall_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(pd.read_csv('../input/raw/atp_matches_' + str(year) + '.csv', parse_dates=['tourney_date']))

    # Concatenate data frames
    matches = pd.concat(matches, sort=False)

    # TODO: implement home advantage, season and climate, need lookup table
    # TODO: implement very recent performance, last month + tournament
    train_columns = ['rel_total_won', 'rel_surface_won', 'mutual_won', 'mutual_surface_won', 'mutual_game', 'rank_diff',
                     'points_grad_diff', 'outcome']
    train_matches = np.zeros((len(matches), len(train_columns)))
    train_matches = pd.DataFrame(mutual_matches, columns=train_columns)

    # Generate training matrix and update statistics matrix
    for i, match in matches.iterrows():
        train_match = train_matches.iloc[i]
        winner_id = match['winner_id']
        loser_id = match['winner_id']
        winner_cond_stats = cond_stats[winner_id]
        loser_cond_stats = cond_stats[loser_id]

        # 1. Relative total won matches differences
        winner_total_won = winner_cond_stats['total_won']
        winner_total_lost = winner_cond_stats['total_lost']
        loser_total_won = loser_cond_stats['total_won']
        loser_total_lost = loser_cond_stats['total_lost']

        winner_rel_total_won = winner_total_won / (winner_total_won + winner_total_lost)
        loser_rel_total_won = loser_total_won / (loser_total_won + loser_total_lost)
        rel_total_won = winner_rel_total_won - loser_rel_total_won
        train_match['rel_total_won'] = rel_total_won

        # 2. Rel surface won differences


        # Update entry
        train_matches.iloc[i] = train_match
