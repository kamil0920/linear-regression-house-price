import pandas as pd
import numpy as np


class NeighbourhoodUtil:

    def get_city_reg(self, state):
        if state in ['Blmngtn', 'Somerst', 'Brookside',
                     'Brookside', 'IDOTRR', 'OldTown',
                     'SWISU']:
            return 'South'
        elif state in ['Blueste', 'CollgCr', 'Edwards',
                       'MeadowV', 'Mitchel', 'Timber']:
            return 'SouthWest'
        elif state in ['Crawfor']:
            return 'SouthEst'
        elif state in ['Blmngtn', 'Somerst', 'BrDale',
                       'ClearCr', 'Gilbert', 'Names', 'NoRidge',
                       'NPkVill', 'NridgHt', 'StoneBr', 'Veenker']:
            return 'North'
        elif state in ['NWAmes', 'Sawyer', 'SawyerW']:
            return 'NorthWest'
        else:
            return 'Missing'

    def get_cardinality(self, data, col):
        index = data[col].value_counts(normalize=True, ascending=True).index
        count = data[col].value_counts(normalize=True, ascending=True).values
        perc = count * 100
        df = pd.DataFrame(index=index, data={'count': count, '%': perc})
        return df

    def bin_categories_to_other(self, df, col, groups, val='Other', delete=True):
        df_ = df.copy()
        new_col_name = col + '_grp'
        df_[new_col_name] = np.where(
            df_[col].isin(groups),
            df_[col].str.title(),
            val)
        if delete == True:
            df_.drop(labels=col, axis=1, inplace=True)
        return df_

