import json
import os

from definitions import GEN_PATH, ROOT_DIR
from statistics import generate_match_statistics
from pre_processing import process_matches

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