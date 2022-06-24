import traceback
from functools import partial

import catboost as ctb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp, space_eval
from hyperopt.pyll import scope
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from preprocess.AnovaTestCategoricalFeature import AnovaTest
from preprocess.SelectorNumericalFeature import FeatureSelector
from preprocess.TransformerNumericalFeature import NumericalFeatureCleaner


class HyperOptimization(object):
    """Tunning hyperperameters with hyperopt."""

    def __init__(self, x_train, x_val, y_train, y_val):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

    def process(self, best_pipe_params, max_evals):
        try:
            space = self.get_estimators()
            objective_func = partial(self.objective_new, dataset_df=self.x_train,
                                     dataset_df_eval=self.x_val, target=self.y_train, target_eval=self.y_val,
                                     best_pipe_params=best_pipe_params)
            trials = Trials()
            result = fmin(fn=objective_func, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

            best_params = space_eval(space, result)
        except Exception as exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return {'status': STATUS_FAIL,
                    'exception': str(exc)}

        return result, trials, best_params

    def objective_new(self, space, dataset_df, dataset_df_eval, target, target_eval, best_pipe_params):
        try:
            model_choice = space['model_choice'][0]
        except Exception as exc:
            model_choice = space['model_choice']

        cat_transformer = best_pipe_params['preprocessor__categorical__cat_trans']

        num_transformer = best_pipe_params['preprocessor__numerical__num_trans__scaler']
        column_transformer = self.make_col_transformer(num_transformer, cat_transformer, dataset_df, target.name)
        X_val_transformed = column_transformer.fit_transform(dataset_df_eval, target_eval)

        model_choice['fit_params']['regressor__eval_set'] = [(X_val_transformed, target_eval)]

        model = self.sample_to_model(model_choice)
        fit_params = self.get_fit_params(model_choice)

        pipe = Pipeline(steps=[
            ('column_transformer', column_transformer),
            ('regressor', model)
        ])
        pipe.fit(dataset_df, target, **fit_params)

        y_predicted = pipe.predict(dataset_df_eval)

        mae = mean_absolute_error(target_eval, y_predicted)
        return {'loss': mae, 'status': STATUS_OK}

    def get_fit_params(self, model_choice):
        fit_params = model_choice['fit_params']
        return fit_params

    def make_col_transformer(self, num_transformer, cat_transformer, X, target):
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        numerical_pipe = self.get_numerical_pipe(num_transformer)
        categorical_pipe = self.get_categorical_pipe(cat_transformer, target)
        column_trans = ColumnTransformer(
            transformers=
            [
                ('num', numerical_pipe, numerical_cols),
                ('cat', categorical_pipe, categorical_cols)
            ]
        )
        return column_trans

    def sample_to_model(self, model_choice):
        model_params = model_choice['params']
        return model_choice['model'](**model_params)

    def get_fature_selector_steps(self):
        estimator = RandomForestRegressor(n_estimators=400, max_depth=10, max_features='auto',
                                          min_samples_leaf=4, min_samples_split=10,
                                          n_jobs=-1, bootstrap=True)

        # Define steps
        step1 = {'Constant Features': {'frac_constant_values': 0.9}}
        step2 = {'Correlated Features': {'correlation_threshold': 0.8}}
        step3 = {'Relevant Features': {'estimator': estimator,
                                       'verbose': 0,
                                       'random_state': 42}}
        step4 = {'RFECV Features': {'cv': 5,
                                    'estimator': estimator,
                                    'step': 1,
                                    'verbose': 0}}

        steps = [step1, step2]
        return steps

    def get_numerical_pipe(self, num_transformer):
        feature_selector_steps = self.get_fature_selector_steps()
        numerical_pipe = Pipeline(steps=[
            ('num_transformator', NumericalFeatureCleaner(num_transformer)),
            ('num_selector', FeatureSelector(feature_selector_steps))
        ])
        return numerical_pipe

    def get_categorical_pipe(self, cat_transformer, target):
        numerical_pipe = Pipeline(steps=[
            # ('cat_selector', AnovaTest(target)),
            ('cat_transformator', cat_transformer)
        ])
        return numerical_pipe

    def get_models(self):
        return {
            'xgb_regressor': xgb.XGBRegressor,
            'lgbm_regressor': lgb.LGBMRegressor,
            'cat_regressor': ctb.CatBoostRegressor,
        }

    def get_estimators(self):
        xgb_regressor = {
                            'model': xgb.XGBRegressor,
                            'params': {
                                'max_depth': scope.int(hp.quniform("xgbregressor__max_depth", 2, 15, 1)),
                                # 'min_child_weight': scope.float(hp.uniform('xgbregressor__min_child_weight', 0.58, 0.63)),
                                # 'colsample_bytree': scope.float(hp.uniform('xgbregressor__colsample_bytree', 0.7, 0.8)),
                                'learning_rate': scope.float(hp.quniform('xgbregressor__learning_rate', 0.09, 0.15, 0.01)),
                                # 'gamma': hp.uniform('xgbregressor__gamma', 1, 2),
                                'n_estimators': scope.int(hp.quniform('xgbregressor__n_estimator', 400, 550, 10))
                            },
                            'fit_params': {
                                'regressor__early_stopping_rounds': 50,
                                'regressor__verbose': False
                            }
                        },
        lgbm_regressor = {
                             'model': lgb.LGBMRegressor,
                             'params': {
                                 'max_depth': scope.int(hp.quniform('lgbmregressor__max_depth', 5, 15, 1)),
                                 # 'min_child_weight': scope.float(hp.uniform('lgbmregressor__min_child_weight', 0.5, 0.7)),
                                 # 'colsample_bytree': scope.float(hp.uniform('lgbmregressor__colsample_bytree', 0.8, 0.9)),
                                 'learning_rate': scope.float(hp.quniform('lgbmregressor__learning_rate', 0.09, 0.16, 0.01)),
                                 # 'subsample': hp.uniform('lgbmregressor__subsample', 0.9, 1),
                                 # 'num_leaves': hp.choice('lgbmregressor__n_estimator__num_leaves',
                                 #                         np.arange(90, 140, 1, dtype=int)),
                                 'n_estimators': scope.int(hp.quniform('lgbmregressor__n_estimator', 400, 500, 10))
                             },
                             'fit_params': {
                                 'regressor__early_stopping_rounds': 50,
                                 'regressor__verbose': False
                             }
                         },

        cat_regressor = {
            'model': ctb.CatBoostRegressor,
            'params': {
                'learning_rate': hp.uniform('ctbregressor__learning_rate', 0.001, 0.5),
                'max_depth': hp.choice('ctbregressor__max_depth', np.arange(2, 15, 1, dtype=int)),
                'colsample_bylevel': hp.choice('ctbregressor__colsample_bylevel',
                                               np.arange(0.3, 0.8, 0.1)),
                'n_estimators': hp.choice('ctbregressor__n_estimator', np.arange(100, 300, 50)),
            },
            'fit_params': {
                'regressor__early_stopping_rounds': 50,
                'regressor__verbose': False
            }
        }

        search_space = {
            'model_choice': hp.choice(
                'model_choice', [xgb_regressor, lgbm_regressor, ]
            )
        }

        return search_space
