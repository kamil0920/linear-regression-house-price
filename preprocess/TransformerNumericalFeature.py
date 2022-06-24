from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd

from scipy.stats import skew


class NumericalFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.remove_outliers(X)
        X = self.skew_data(X)
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return X

    def remove_outliers(self, X):
        for col in X.columns:
            Q1 = X[col].quantile(0.05)
            Q3 = X[col].quantile(0.95)
            X[col] = np.where(X[col] < Q1, Q1, X[col])
            X[col] = np.where(X[col] > Q3, Q3, X[col])
        return X

    def isolation_forest(self, X):
        # https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2
        to_model_columns = X.columns
        clf = IsolationForest(n_estimators=100, max_samples='auto',
                              max_features=0.9, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
        clf.fit(X[to_model_columns])
        pred = clf.predict(X[to_model_columns])
        X['anomaly'] = pred
        outliers = X.loc[X['anomaly'] == -1]
        outlier_index = list(outliers.index)
        X = X.drop([outlier_index])
        return X

    def skew_data(self, X):
        # https://www.hackerearth.com/practice/machine-learning/machine-learning-projects/python-project/tutorial/
        skewed = X.apply(lambda x: skew(x))
        skewed = skewed[skewed > 0.75]
        skewed = skewed.index
        X[skewed] = np.log1p(X[skewed])
        return X
