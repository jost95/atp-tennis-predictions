# Helper functions
import datetime
import pandas as pd
import numpy as np
import os
import re
from definitions import RAW_PATH


def get_home_advantage(winner_ioc, loser_ioc, tourneys, tourney_name):
    location = filter_tourney_name(tourney_name)
    country_code = tourneys.loc[tourneys.location == location, 'country_code']

    if len(country_code) == 0:
        return 0
    else:
        country_code = country_code.iloc[0]

        if winner_ioc == country_code and loser_ioc == country_code:
            return 0
        elif winner_ioc == country_code:
            return 1
        elif loser_ioc == country_code:
            return -1
        else:
            return 0


def filter_tourney_name(words):
    # This is not the most elegant solution but it works
    s = ' '.join(
        w for w in words.split() if (len(w) > 3 or (len(w) == 3 and w[len(w) - 1].isalpha()) or w[len(w) - 1] == '.'))

    # Make lowercase
    s = s.lower()

    # Remove anything but letters and space
    s = re.sub(r'[^a-z\s]', u'', s, flags=re.UNICODE)

    # Replace all davis cup
    if 'davis cup' in s:
        s = ''

    return s


def fetch_country(location, session):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": location,
        "format": "json",
        "addressdetails": 1,
        "accept-language": "en-US"
    }

    results = session.get(url=url, params=params).json()

    # If no search results found (e.g. 'atp challenger tour finals')
    if len(results) == 0:
        country = location
    else:
        country = results[0]['address']['country']

    return country


def extract_player_ids(years):
    # Extract active players in a specific year range

    matches = load_matches(years)
    winner_ids = matches.winner_id.to_numpy()
    loser_ids = matches.loser_id.to_numpy()

    # Filter out unique players
    players = np.unique(np.append(winner_ids, loser_ids))

    print('Players loaded, number of players:', len(players))
    return players


def load_matches(years, player_ids=None):
    # Load matches in a specific year range
    # If specified, sorts out matches where no players are in player_ids
    matches = []

    for year in range(years['from'], years['to'] + 1):
        matches.append(
            pd.read_csv(os.path.join(RAW_PATH, 'atp_matches_futures_' + str(year) + '.csv'),
                        parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv(os.path.join(RAW_PATH, 'atp_matches_qual_chall_' + str(year) + '.csv'),
                        parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv(os.path.join(RAW_PATH, 'atp_matches_' + str(year) + '.csv'), parse_dates=['tourney_date']))

    matches = pd.concat(matches, sort=False)

    if player_ids is not None:
        # Remove not wanted matches
        matches = matches[matches['winner_id'].isin(player_ids) | matches['loser_id'].isin(player_ids)]

    # Drop not relevant columns
    matches = matches.filter(
        ['tourney_name', 'winner_id', 'winner_ioc', 'loser_id', 'loser_ioc', 'tourney_date', 'tourney_level',
         'surface', 'score', 'match_num', 'tourney_id'])

    # Sort by date (oldest ranking first)
    matches.sort_values(by=['tourney_date', 'match_num'], inplace=True, ascending=True)

    print('Matches loaded, number of matches:', len(matches))
    return matches


def get_time_weight(base_year, current_date, sign=1):
    # Return an exponential weighted time decay
    # Pay attention to value of sign
    time_delta = (datetime.date(base_year + 1, 1, 1) - current_date.date()).days
    return np.exp(-sign * time_delta / (365 * 3))


def get_surface(surface):
    # Guesses the surface as hard if specified is not known
    surface = str(surface).lower()
    return 'hard' if surface == 'nan' or surface == 'none' else surface


def get_score(score):
    # Extracts the score in games from the score string
    score = str(score)

    # Continue if score is not known
    if score == 'nan':
        return 0, 0

    sets = score.split()
    winner_games = 0
    loser_games = 0

    for s in sets:
        # Remove weird scoring
        games = ''.join(c for c in s if c not in '[]RET').split('-')

        if len(games) == 1:
            continue

        # This might cause trouble if score not properly formatted
        winner_games += int(games[0][0])
        loser_games += int(games[1][0])

    return winner_games, loser_games


def print_progress(i, no_matches):
    # Prints the process
    if i % 10000 == 0:
        print(i, 'matches (' + str(round(i / no_matches * 100, 2)) + '%) processed')


def load_rankings():
    # Loads player rankings and sorts them in ascending order
    rankings_10s = pd.read_csv(os.path.join(RAW_PATH, 'atp_rankings_10s.csv'), parse_dates=['ranking_date'])
    rankings_current = pd.read_csv(os.path.join(RAW_PATH, 'atp_rankings_current.csv'), parse_dates=['ranking_date'])
    rankings = pd.concat([rankings_10s, rankings_current], sort=False)

    # Sort by date (oldest ranking first)
    rankings.sort_values(by=['ranking_date'], inplace=True, ascending=True)

    print('Rankings loaded')
    return rankings


def get_tourney_games(winner_id, loser_id, recent_matches, tourney_id, match_num):
    # Get recent performance in relative number of games IN CURRENT tournament
    games_won_winner = 0
    gammes_played_winner = 0
    games_won_loser = 0
    games_played_loser = 0

    for match in recent_matches.itertuples():
        if match.tourney_id == tourney_id and match.match_num < match_num:
            winner_games, loser_games = get_score(match.score)

            if match.winner_id == winner_id:
                games_won_winner += winner_games
                gammes_played_winner += winner_games
            elif match.winner_id == loser_id:
                games_won_loser += loser_games
                games_played_loser += loser_games

            if match.loser_id == winner_id:
                gammes_played_winner += loser_games
            elif match.loser_id == loser_id:
                games_played_loser += loser_games

    if gammes_played_winner == 0:
        rel_wins_winner = 0
    else:
        rel_wins_winner = games_won_winner / gammes_played_winner

    if games_played_loser == 0:
        rel_wins_loser = 0
    else:
        rel_wins_loser = games_won_loser / games_played_loser

    return rel_wins_winner - rel_wins_loser


def get_recent_performance(winner_id, loser_id, recent_matches, tourney_id):
    # Extract recent perfomance in terms of relative win from recent matches BEFORE tournament
    recent_wins_winner = 0
    recent_played_winner = 0
    recent_wins_loser = 0
    recent_played_loser = 0

    for match in recent_matches.itertuples():
        if match.tourney_id != tourney_id:
            if match.winner_id == winner_id:
                recent_wins_winner += 1
                recent_played_winner += 1
            elif match.winner_id == loser_id:
                recent_wins_loser += 1
                recent_played_loser += 1

            if match.loser_id == winner_id:
                recent_played_winner += 1
            elif match.loser_id == loser_id:
                recent_played_loser += 1

    if recent_played_winner == 0:
        rel_wins_winner = 0
    else:
        rel_wins_winner = recent_wins_winner / recent_played_winner

    if recent_played_loser == 0:
        rel_wins_loser = 0
    else:
        rel_wins_loser = recent_wins_loser / recent_played_loser

    return rel_wins_winner - rel_wins_loser


def get_relative_climate_wins(cond_stats, winner_id, loser_id, climate):
    # For each player, calculates the ratio won matches in the current climate and
    # then takes the difference between them
    climate_wins_winner = cond_stats['climate_' + climate + '_wins'][winner_id]
    climate_losses_winner = cond_stats['climate_' + climate + '_losses'][winner_id]
    climate_played_winner = climate_wins_winner + climate_losses_winner
    climate_wins_loser = cond_stats['climate_' + climate + '_wins'][loser_id]
    climate_losses_loser = cond_stats['climate_' + climate + '_losses'][loser_id]
    climate_played_loser = climate_wins_loser + climate_losses_loser

    if climate_played_winner == 0:
        rel_climate_wins_winner = 0
    else:
        rel_climate_wins_winner = float(climate_wins_winner) / climate_played_winner

    if climate_played_loser == 0:
        rel_climate_wins_loser = 0
    else:
        rel_climate_wins_loser = float(climate_wins_loser) / climate_played_loser

    return rel_climate_wins_winner - rel_climate_wins_loser


def get_relative_surface_wins(cond_stats, winner_id, loser_id, surface):
    # For each player, calculates the ratio won matches on the surface and
    # then takes the difference between them
    surface_wins_winner = cond_stats['surface_' + surface + '_wins'][winner_id]
    surface_losses_winner = cond_stats['surface_' + surface + '_losses'][winner_id]
    surface_played_winner = surface_wins_winner + surface_losses_winner
    surface_wins_loser = cond_stats['surface_' + surface + '_wins'][loser_id]
    surface_losses_loser = cond_stats['surface_' + surface + '_losses'][loser_id]
    surface_played_loser = surface_wins_loser + surface_losses_loser

    if surface_played_winner == 0:
        rel_surface_wins_winner = 0
    else:
        rel_surface_wins_winner = float(surface_wins_winner) / surface_played_winner

    if surface_played_loser == 0:
        rel_surface_wins_loser = 0
    else:
        rel_surface_wins_loser = float(surface_wins_loser) / surface_played_loser

    return rel_surface_wins_winner - rel_surface_wins_loser


def get_relative_total_wins(cond_stats, winner_id, loser_id):
    # For each player, calculates the ratio won matches in total and
    # then takes the difference between them
    total_wins_winner = cond_stats['total_wins'][winner_id]
    total_losses_winner = cond_stats['total_losses'][winner_id]
    total_played_winner = total_wins_winner + total_losses_winner
    total_wins_loser = cond_stats['total_wins'][loser_id]
    total_losses_loser = cond_stats['total_losses'][loser_id]
    total_played_loser = total_wins_loser + total_losses_loser

    if total_played_winner == 0:
        rel_total_wins_winner = 0
    else:
        rel_total_wins_winner = float(total_wins_winner) / total_played_winner

    if total_played_loser == 0:
        rel_total_wins_loser = 0
    else:
        rel_total_wins_loser = float(total_wins_loser) / total_played_loser

    return rel_total_wins_winner - rel_total_wins_loser


def get_mutual_surface_wins(mm_clay, mm_grass, mm_hard, surface, winner_id, loser_id):
    # Calculates the difference in wins between opponents on specified surface
    if surface == 'clay':
        return mm_clay[winner_id][loser_id] - mm_clay[loser_id][winner_id]
    elif surface == 'grass':
        return mm_grass[winner_id][loser_id] - mm_grass[loser_id][winner_id]
    else:
        return mm_hard[winner_id][loser_id] - mm_hard[loser_id][winner_id]


def get_rankings(rankings, winner_id, loser_id, tourney_date):
    # Get the current ranking differents and the one year ranking gradient in points
    winner_rankings = rankings.loc[rankings['player'] == winner_id]
    loser_rankings = rankings.loc[rankings['player'] == loser_id]
    highest_numbered_ranking = np.max(rankings['rank'])

    # Set date as ranking index, delete possible duplicates due to overlapping lists
    winner_rankings.set_index('ranking_date', inplace=True)
    winner_rankings = winner_rankings.loc[~winner_rankings.index.duplicated(keep='first')]
    loser_rankings.set_index('ranking_date', inplace=True)
    loser_rankings = loser_rankings.loc[~loser_rankings.index.duplicated(keep='first')]

    # It is not certain that all players have a ranking right now
    try:
        winner_current_rank = winner_rankings.iloc[winner_rankings.index.get_loc(tourney_date, method='pad')]
        winner_current_points = winner_current_rank['points']
        winner_current_rank = winner_current_rank['rank']
    except KeyError:
        winner_current_rank = highest_numbered_ranking + 1
        winner_current_points = 0

    try:
        loser_current_rank = loser_rankings.iloc[loser_rankings.index.get_loc(tourney_date, method='pad')]
        loser_current_points = loser_current_rank['points']
        loser_current_rank = loser_current_rank['rank']
    except KeyError:
        loser_current_rank = highest_numbered_ranking + 1
        loser_current_points = 0

    rank_diff = winner_current_rank - loser_current_rank

    last_year_date = tourney_date - pd.DateOffset(years=1)

    # It is not certain that all players had a ranking one year ago
    try:
        winner_old_rank = winner_rankings.iloc[winner_rankings.index.get_loc(last_year_date, method='pad')]
        winner_old_points = winner_old_rank['points']
    except KeyError:
        winner_old_points = 0

    try:
        loser_old_rank = winner_rankings.iloc[winner_rankings.index.get_loc(last_year_date, method='pad')]
        loser_old_points = loser_old_rank['points']
    except KeyError:
        loser_old_points = 0

    winner_points_grad = winner_current_points - winner_old_points
    loser_points_grad = loser_current_points - loser_old_points
    points_grad_diff = winner_points_grad - loser_points_grad

    return rank_diff, points_grad_diff
