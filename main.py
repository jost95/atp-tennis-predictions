import json
import os

import pandas as pd

from definitions import GEN_PATH, ROOT_DIR
from utilities.statistics import generate_match_statistics
from utilities.pre_processing import process_matches
from utilities.feature_selection import evaluate_features

# Read configuration file
with open(os.path.join(ROOT_DIR, 'config.json')) as f:
    config = json.load(f)

stats_filepath = os.path.join(GEN_PATH, config['stats_filename'])
proc_match_filepath = os.path.join(GEN_PATH, config['proc_match_filename'])
base_weight = config['base_weight']
t_weights = config['tourney_weights']
t_levels = config['tourney_levels']
stats_years = config['stats_year']
proc_years = config['proc_year']

# GENERATE STATISTICS
# - create new statistical data to be used for training
if config['generate_stats']:
    generate_match_statistics(stats_filepath, t_weights, base_weight, stats_years, proc_years)

# FEATURE ENGINEERING
# - generate new features to be evaluated
if config['generate_training']:
    process_matches(stats_filepath, proc_match_filepath, t_weights, base_weight, proc_years, t_levels)

# Load engineered data set
data = pd.read_hdf(proc_match_filepath, key='matches')

# Split data set into train and test
# Start of 2019 season in unix time, 2018-12-31 00:00:00 UTC
test_date_begin = 1546214400
# change to date to tourney_date and add tourney_level
drop_cols = ['outcome', 'date']

data_test = data.loc[data.date >= test_date_begin]
y_test = data_test.outcome
# t_level_test = data_test.tourney_level
X_test = data_test.drop(drop_cols, axis=1)

# Final testing will be done on 2019 matches
data_train = data.loc[data.date < test_date_begin]
y_train = data_train.outcome
X_train = data_train.drop(drop_cols, axis=1)

# FEATURE SELECTION/EVALUATION
# - show feature selection results
if config['evaluate_features']:
    evaluate_features(X_train, y_train)

# Based on feature selection results, drop features
drop_features = ['rel_total_wins', 'mutual_wins', 'rel_climate_wins']
X_train = X_train.drop(drop_features, axis=1)
X_test = X_test.drop(drop_features, axis=1)

# MODEL SELECTION
# Use cross-validation to split training data into training and validation sets

# Evaluate implemented models on set
# 1. kNN
# 2. Tree-based
# 4. SVM
# 5. Neural Network

# Evaluate selection test set, report results and most importantly score
