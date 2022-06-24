from sklearn.base import TransformerMixin, BaseEstimator


class BinningFeature(BaseEstimator, TransformerMixin):
    def __init__(self, factor=0.08):
        self.factor = factor

    def binning_low_fraction_categories(self, X):
        each_categories_fraction = X.value_counts() / len(X)
        X = X.astype('category')
        if 'Other' not in each_categories_fraction.index.to_list():
            X = X.cat.add_categories(['Other'])
        for idx, value in each_categories_fraction.iteritems():
            if value < self.factor:
                X.where(X == idx, 'Other')
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.binning_low_fraction_categories)