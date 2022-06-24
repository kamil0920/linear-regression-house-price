import numpy as np
import pandas as pd


class GetModelResults(object):

    def __init__(self, results_df):
        self.results_df = results_df

    def get_model_results(self, opt, model_opt):
        trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(self.unpack) for t in opt.trials])
        trials_df["loss"] = [t["result"]["loss"] for t in opt.trials]
        trials_df["trial_number"] = trials_df.index
        # trials_df['model_choice'] = trials_df['model_choice'].apply(lambda x: 'xgb_regressor' if x == 0 else 'lgbm_regressor')
        return trials_df.loc[trials_df['model_choice'] == model_opt].dropna(axis=1)

    def unpack(self, X):
        if X:
            return X[0]
        return np.nan
