from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import numpy as np
import pandas as pd


class CustomIterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='ball_tree'):
        self.cat_cols = []
        self.num_cols = []
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm

    def fit(self, X, y=None):
        self.get_column_metadata(X)
        return self

    def get_column_metadata(self, X):
        """Fit column selector, get names and indicies for categorical and numerical columns."""
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns
        self.num_cols = X.select_dtypes(include=[np.number]).columns

    def transform(self, X):
        """Takes in X dataframe unprocessed features and rerturns
        dataframe without missing values, either imputed using MICE 
        method or constant imputation, depending on proportion of missing values
        in a column."""
        X_ = X.copy()
        impute_mask = self.get_impute_mask(X_)
        X_no_nan = self.replace_nan_const(X_)
        X_cat_numercalized = self.apply_ordinal_encoder(X_no_nan[self.cat_cols])

        X_numercalized = np.hstack((X_cat_numercalized, X_no_nan[self.num_cols]))
        X_fill_back_nan = self.fill_back_nan(X_numercalized, impute_mask)
        X_imputed = self.apply_imputer(X_fill_back_nan)

        cat_cols_array = np.array(self.cat_cols)
        num_cols_array = np.array(self.num_cols)

        df = pd.DataFrame(self.invert_cat_encoder(X_imputed), columns=np.concatenate((num_cols_array, cat_cols_array)))

        df[self.cat_cols] = df[self.cat_cols].astype('category')
        df[self.num_cols] = df[self.num_cols].astype('int32')

        return df

    def get_impute_mask(self, X):
        """Get boolean mask marking value locations that need to be iteratively imputed.
        Only impute those columns, where proportion of missing values is <50%.
        Otherwise leave constant imputation."""
        cols_most_values_missing = [col for col in X.columns if X[col].isnull().sum() / X.shape[0] > .5]

        impute_mask = X.isnull()
        impute_mask[cols_most_values_missing] = False
        return impute_mask

    def replace_nan_const(self, X):
        """Use fitted ColumnSelector to get categorical and numerical column names.
        Fill missing values with 'None' and zero constant accordingly."""

        X[self.cat_cols] = X[self.cat_cols].astype('string')
        X[self.cat_cols] = X[self.cat_cols].fillna('None')
        X[self.cat_cols] = X[self.cat_cols].astype('category')
        X[self.num_cols] = X[self.num_cols].fillna(0)
        return X

    def apply_ordinal_encoder(self, X_no_nan_cat):
        """Apply Ordinal encoder for categorical column,
        to get integer calues for all categories, iuncluding missing.
        Make encoder available on class scope, for inversion later."""
        self.ordinal_encoder = OrdinalEncoder()
        X_cat_inverted = self.ordinal_encoder.fit_transform(X_no_nan_cat)
        return X_cat_inverted

    def fill_back_nan(self, X_numercalized, impute_mask):
        """Replace back constant values with nan's, according to imputation mask."""
        X_numercalized[impute_mask] = np.nan
        return X_numercalized

    def apply_imputer(self, X_fill_back_nan):
        """Use IterativeImputer to predict missing values."""
        imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                                 weights=self.weights,
                                                                 algorithm=self.algorithm),
                                   random_state=42
                                   )
        transform = imputer.fit_transform(X_fill_back_nan)
        return transform

    def invert_cat_encoder(self, X_imputed):
        """Invert ordinal encoder to  get back categorical values"""
        X_cats = X_imputed[:, :len(self.cat_cols)]
        X_cat_inverted = self.ordinal_encoder.inverse_transform(X_cats)
        X_numercs = X_imputed[:, len(self.cat_cols):]
        return np.hstack((X_numercs, X_cat_inverted))
