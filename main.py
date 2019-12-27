import pandas as pd
from utilities.statistics import generate_match_statistics
from utilities.pre_processing import generate_data

# Weights
base_weight = 100
t_weights = {'S': 0.125, 'C': 0.25, 'A': 0.5, 'M': 0.75, 'D': 0.75, 'G': 1, 'F': 1}

# Stats phase
generate_statistics = False
from_stats_year = 2010
to_stats_year = 2014
from_data_year = to_stats_year + 1
to_data_year = 2019

if generate_statistics:
    generate_match_statistics(t_weights, base_weight, from_stats_year, to_stats_year, from_data_year, to_data_year)

# Generate training and testing data
generate_data(t_weights, base_weight, from_data_year, to_data_year)

