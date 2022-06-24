from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import scipy.stats as stats

import numpy as np


class AnovaTest(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.cat_cols = []
        self.target = target

    def one_way_anova(self, X, column, target):

        data = X[[column, target]]

        groups = data.groupby(column).groups
        data_pivot = pd.pivot_table(data, values=target, columns=column)

        list_new = []
        for col in data_pivot.columns:
            list_new.append(data.iloc[groups[col]][target].dropna())
        return stats.f_oneway(*list_new)

    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X[self.target] = y
        for col in X.select_dtypes(exclude=[np.number]).columns:
            if X[col].unique().size < 3:
                self.cat_cols.append(col)
            else:
                f_val, p_val = self.one_way_anova(X, col, self.target)
                if p_val < 0.05:
                    self.cat_cols.append(col)
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.cat_cols]
