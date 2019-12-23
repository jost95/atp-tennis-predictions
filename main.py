import pandas as pd
from utilities import calc_stats
from utilities import gen_train

# Base weight for matches
base_weight = 10

# Stats phase
generate_statistics = False
from_stats_year = 2010
to_stats_year = 2014

if generate_statistics:
    calc_stats.calc_stats(base_weight, from_stats_year, to_stats_year)

# Training phase
from_train_year = to_stats_year + 1
to_train_year = 2018
gen_train.gen_train(base_weight, from_train_year, to_train_year)

