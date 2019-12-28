import pandas as pd
from utilities.statistics import generate_match_statistics
from utilities.pre_processing import generate_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
t_points = {'A': 23, 'M': 15}

# Select years (including specified year)
from_stats_year = 2010
to_stats_year = 2014
from_data_year = to_stats_year + 1
to_data_year = 2019

generate_statistics = False

# Create new statistical data to be used for training
if generate_statistics:
    generate_match_statistics(t_weights, base_weight, from_stats_year, to_stats_year, from_data_year, to_data_year)

generate_training = False

# Generate new training data to be evaluated
if generate_training:
    generate_data(t_weights, base_weight, from_data_year, to_data_year)

# Load data set
data = pd.read_hdf('input/generated/processed_matches.h5', key='matches')

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
