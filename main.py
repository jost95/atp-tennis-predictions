import json
import os

import pandas as pd

from definitions import GEN_PATH, ROOT_DIR
from utilities.statistics import generate_match_statistics
from utilities.pre_processing import process_matches
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read configuration file
with open(os.path.join(ROOT_DIR, 'config.json')) as f:
    config = json.load(f)

stats_filepath = os.path.join(GEN_PATH, config['stats_filename'])
proc_match_filepath = os.path.join(GEN_PATH, config['proc_match_filename'])
base_weight = config['base_weight']
t_weights = config['tourney_weights']
stats_years = config['stats_year']
proc_years = config['proc_year']

# Create new statistical data to be used for training
if config['generate_statistics']:
    generate_match_statistics(stats_filepath, t_weights, base_weight, stats_years, proc_years)

# Generate new training data to be evaluated
if config['generate_training']:
    process_matches(stats_filepath, proc_match_filepath, t_weights, base_weight, proc_years)

# Load data set
data = pd.read_hdf(proc_match_filepath, key='matches')

y = data.outcome
X = data.drop('outcome', axis=1)

# Split should be made with more thought first

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Do kNN algorithm
k = round(len(X_train) ** 0.5)

knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(accuracy_score(y_test, predictions))
