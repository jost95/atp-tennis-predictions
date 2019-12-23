import pandas as pd
import numpy as np


def gen_data(base_weight, from_year, to_year):
    # Read processed data
    mutual_matches = pd.read_csv('../input/fixed/mutual_matches_to_' + str(to_year - 1) + '.csv')
    mutual_matches_clay = pd.read_csv('../input/fixed/mutual_matches_clay_to_' + str(to_year - 1) + '.csv')
    mutual_matches_grass = pd.read_csv('../input/fixed/mutual_matches_grass_to_' + str(to_year - 1) + '.csv')
    mutual_matches_hard = pd.read_csv('../input/fixed/mutual_matches_hard_to_' + str(to_year - 1) + '.csv')
    mutual_games = pd.read_csv('../input/fixed/mutual_games_to_' + str(to_year - 1) + '.csv')
    cond_stats = pd.read_csv('../input/fixed/cond_stats_to_' + str(to_year - 1) + '.csv')

    # Read rankings
    rankings_10s = pd.read_csv('../input/raw/atp_rankings_10s.csv', parse_dates=['ranking_date'])

    # Column name needs to be added to below to be read properly
    rankings_current = pd.read_csv('../input/raw/atp_rankings_current.csv', parse_dates=['ranking_date'])

    rankings = pd.concat([rankings_10s, rankings_current], sort=False)

    # Sort by date (oldest ranking first)
    rankings.sort_values(by=['ranking_date'], inplace=True, ascending=True)

    # Read training matches
    matches = []

    for year in range(from_year, to_year + 1):
        matches.append(
            pd.read_csv('../input/raw/atp_matches_futures_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv('../input/raw/atp_matches_qual_chall_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(pd.read_csv('../input/raw/atp_matches_' + str(year) + '.csv', parse_dates=['tourney_date']))

    # Concatenate data frames
    matches = pd.concat(matches, sort=False)

    # Sort by date (oldest first)
    matches.sort_values(by=['tourney_date'], inplace=True, ascending=True)

    # TODO: implement home advantage, season and climate, need lookup table
    # TODO: implement very recent performance, last month + tournament
    train_columns = ['rel_total_won', 'rel_surface_won', 'mutual_won', 'mutual_surface_won', 'mutual_game', 'rank_diff',
                     'points_grad_diff', 'outcome']
    train_matches = np.zeros((len(matches), len(train_columns)))
    train_matches = pd.DataFrame(train_matches, columns=train_columns)

    # Generate training matrix and update statistics matrix
    for i, match in matches.iterrows():
        train_match = train_matches.iloc[i]
        winner_id = match['winner_id']
        loser_id = match['winner_id']

        # Extract conditional stats
        winner_cond_stats = cond_stats[winner_id]
        loser_cond_stats = cond_stats[loser_id]

        # Guess surface as hard if not known
        surface = str(match['surface']).lower()
        surface = 'hard' if surface == 'nan' or surface == 'none' else surface

        # 1. Relative total won matches differences
        winner_total_won = winner_cond_stats['total_won']
        winner_total_lost = winner_cond_stats['total_lost']
        loser_total_won = loser_cond_stats['total_won']
        loser_total_lost = loser_cond_stats['total_lost']

        winner_rel_total_won = winner_total_won / (winner_total_won + winner_total_lost)
        loser_rel_total_won = loser_total_won / (loser_total_won + loser_total_lost)
        rel_total_won = winner_rel_total_won - loser_rel_total_won
        train_match['rel_total_won'] = rel_total_won

        # 2. Relative surface won differences
        winner_surface_won = winner_cond_stats['surface_' + surface + '_won']
        winner_surface_lost = winner_cond_stats['surface_' + surface + '_lost']
        loser_surface_won = loser_cond_stats['surface_' + surface + '_won']
        loser_surface_lost = loser_cond_stats['surface_' + surface + '_won']

        winner_rel_surface_won = winner_surface_won / (winner_surface_won + winner_surface_lost)
        loser_rel_surface_won = loser_surface_won / (loser_surface_won + loser_surface_lost)
        rel_surface_won = winner_rel_surface_won - loser_rel_surface_won
        train_match['rel_surface_won'] = rel_surface_won

        # 3. Mutual won
        mutual_won = mutual_matches[winner_id][loser_id] - mutual_matches[loser_id][winner_id]
        train_match['mutual_won'] = mutual_won

        # 4. Mutual surface won
        # Extract win on surface
        if surface == 'clay':
            mutual_surface_won = mutual_matches_clay[winner_id][loser_id] - mutual_matches_clay[loser_id][winner_id]
        elif surface == 'grass':
            mutual_surface_won = mutual_matches_grass[winner_id][loser_id] - mutual_matches_grass[loser_id][winner_id]
        else:
            mutual_surface_won = mutual_matches_hard[winner_id][loser_id] - mutual_matches_hard[loser_id][winner_id]

        train_match['mutual_surface_won'] = mutual_surface_won

        # 4. Mutual game
        mutual_game = mutual_games[winner_id][loser_id] - mutual_games[loser_id][winner_id]
        train_match['mutual_game'] = mutual_game

        # 5. Rank diff
        winner_rankings = rankings.loc[rankings['player'] == winner_id]
        loser_rankings = rankings.loc[rankings['player'] == loser_id]

        # Find current ranking from week before
        tourney_date = match['tourney_date']
        winner_current_rank = winner_rankings.iloc(winner_rankings.index.get_loc(tourney_date, method='pad'))
        loser_current_rank = loser_rankings.iloc(loser_rankings.index.get_loc(tourney_date, method='pad'))

        

        last_year_date = tourney_date - pd.DateOffset(years=1)
        winner_old_rank = winner_rankings.iloc(winner_rankings.index.get_loc(last_year_date, method='pad'))
        loser_old_rank = winner_rankings.iloc(winner_rankings.index.get_loc(last_year_date, method='pad'))

        # Update entry
        train_matches.iloc[i] = train_match
