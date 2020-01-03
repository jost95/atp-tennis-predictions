# For exploring and selecting features
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE


def evaluate_features(X, y):
    # Declare base model for evaluation
    model = ExtraTreesClassifier()
    model.fit(X, y)

    # 1. Feature correlation
    # Leave out categorical home_advantage variable
    X_heat = X.drop('home_advantage')
    sn.heatmap(X_heat.corr(), annot=True)
    plt.show()

    # 2. Feature importantace
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    # 3. Recursive feature elimination
    rfe_model = RFE(model)
    rfe_model.fit(X, y)
    print("Number of features: %d" % fit.n_features_)
    print("Selected features: %s" % fit.support_)
    print("Feature ranking: %s" % fit.ranking_)
