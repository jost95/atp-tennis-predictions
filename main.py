from utilities.statistics import generate_match_statistics
from utilities.pre_processing import generate_data

base_weight = 100  # so integers can be used in data matrices

# Tournament weights
# S = Satellite/Futures
# C = Challenger/Qualifying
# A = ATP 250/500
# M = ATP 1000
# D = Davis Cup
# G = Grand Slam
# F = ATP Finals
t_weights = {'S': 0.125, 'C': 0.25, 'A': 0.5, 'M': 0.75, 'D': 0.75, 'G': 1, 'F': 1}

# Select years (including specified year)
from_stats_year = 2010
to_stats_year = 2014
from_data_year = to_stats_year + 1
to_data_year = 2019

generate_statistics = True

# Create new statistical data to be used for training
if generate_statistics:
    generate_match_statistics(t_weights, base_weight, from_stats_year, to_stats_year, from_data_year, to_data_year)

generate_training = True

# Generate new training data to be evaluated
if generate_training:
    generate_data(t_weights, base_weight, from_data_year, to_data_year)

# Algorithms
