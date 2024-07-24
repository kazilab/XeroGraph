import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.exceptions import ConvergenceWarning
from sklearn import impute
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from statsmodels.imputation import mice
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel
from .xputer_main import Xpute
# Ignore ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class XeroCompare:
    """
    Initialize the DataAnalyzer with data.

    Parameters:
    data (DataFrame): A pandas DataFrame with potential missing values.
    """
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The data must be a pandas DataFrame.")

        missing_value_indicators = ['NAN', 'Nan', 'NA#', '#VALUE!', '#DIV/0!',
                                    '-', '_', '--', '---', '__', '___', '#N/A',
                                    '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
                                    '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>',
                                    'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a',
                                    'nan', 'null', '#NUM!']

        # Replace all missing value indicators with np.nan, update the df inplace
        df = df.replace(missing_value_indicators, np.nan)

        # Remove rows with missing values to create test_data
        self.ori_data = df.dropna()
        test_data = df.dropna()

        # Introduce missing values randomly 10-20%
        for col in test_data.columns:
            missing_indices = test_data.sample(frac=np.random.uniform(0.1, 0.2)).index
            test_data.loc[missing_indices, col] = np.nan

        self.data = test_data

    def compare(self, run_mice=False):
        if run_mice:
            methods = {
                'mean': mean_imputation(self.data),
                'median': median_imputation(self.data),
                'most_frequent': most_frequent_imputation(self.data),
                'KNN': knn_imputation(self.data),
                'iterative': iterative_imputation(self.data),
                'random_forest': random_forest_imputation(self.data),
                'lasso_cv': lasso_cv_imputation(self.data),
                'xgboost': xgboost_imputation(self.data),
                'xputer': xputer_imputation(self.data),
                'mice': mice_imp(self.data)
            }
        else:
            methods = {
                'mean': mean_imputation(self.data),
                'median': median_imputation(self.data),
                'most_frequent': most_frequent_imputation(self.data),
                'KNN': knn_imputation(self.data),
                'iterative': iterative_imputation(self.data),
                'random_forest': random_forest_imputation(self.data),
                'lasso_cv': lasso_cv_imputation(self.data),
                'xgboost': xgboost_imputation(self.data),
                'xputer': xputer_imputation(self.data)
            }
        results = []
        for method, imp_data in methods.items():
            rmse_scores = []
            p_values = []
            for col in self.data.columns:
                original = self.ori_data[col].values
                imputed = imp_data[col].values
                valid_mask = np.isnan(self.data[col].values)
                rmse = np.sqrt(mean_squared_error(original[valid_mask], imputed[valid_mask]))
                if len(original[valid_mask]) > 1:
                    _, p_value = ttest_rel(original[valid_mask], imputed[valid_mask])
                else:
                    p_value = np.nan
                rmse_scores.append(rmse)
                p_values.append(p_value)
            rmse_mean = np.mean(rmse_scores)
            rmse_median = np.median(rmse_scores)
            rmse_min = np.min(rmse_scores)
            rmse_max = np.max(rmse_scores)
            p_mean = np.mean(p_values)
            p_median = np.median(p_values)
            p_min = np.min(p_values)
            p_max = np.max(p_values)
            results.append({'Method': method, 'RMSE-mean': rmse_mean, 'RMSE-median': rmse_median,
                            'RMSE-min': rmse_min, 'RMSE-max': rmse_max, 'P-mean': p_mean,
                            'P-median': p_median, 'P-min': p_min, 'P-max': p_max})

        summary = pd.DataFrame(results)
        return summary


def mean_imputation(data):
    print("Performing mean imputation")
    imputer = impute.SimpleImputer(strategy='mean')
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def median_imputation(data):
    print("Performing median imputation")
    imputer = impute.SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def most_frequent_imputation(data):
    print("Performing most frequent imputation")
    imputer = impute.SimpleImputer(strategy='most_frequent')
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def knn_imputation(data):
    print("Performing KNN imputation")
    imputer = impute.KNNImputer()
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def iterative_imputation(data, max_iter=25):
    print("Performing Iterative imputation")
    min_value = np.min(np.min(data, axis=0))
    max_value = np.max(np.max(data, axis=0))
    imputer = impute.IterativeImputer(max_iter=max_iter, random_state=0,
                                      min_value=min_value, max_value=max_value)
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def random_forest_imputation(data):
    print("Performing Iterative imputation using Random Forest")
    min_value = np.min(np.min(data, axis=0))
    max_value = np.max(np.max(data, axis=0))
    imputer = impute.IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1), max_iter=100, random_state=0,
                                      min_value=min_value, max_value=max_value)
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def lasso_cv_imputation(data):
    print("Performing Iterative imputation using Lasso CV")
    min_value = np.min(np.min(data, axis=0))
    max_value = np.max(np.max(data, axis=0))
    imputer = impute.IterativeImputer(estimator=LassoCV(max_iter=100000, n_jobs=-1), max_iter=1000, random_state=0,
                                      min_value=min_value, max_value=max_value)
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def xgboost_imputation(data):
    print("Performing Iterative imputation using XGBoost")
    min_value = np.min(np.min(data, axis=0))
    max_value = np.max(np.max(data, axis=0))
    imputer = impute.IterativeImputer(estimator=XGBRegressor(n_jobs=-1), max_iter=100, random_state=0,
                                      min_value=min_value, max_value=max_value)
    imputed = imputer.fit_transform(data)
    df = pd.DataFrame(imputed, index=data.index, columns=data.columns)
    return df


def xputer_imputation(data):
    print("Performing imputation using Xputer")
    imputer = Xpute(impute_zeros=False, pre_imputation='MixType', xgb_models=3, mf_for_xgb=True,
                    use_transformed_df=False, optuna_for_xgb=True, optuna_n_trials=50, n_iterations=3,
                    save_imputed_df=False, save_plots=False, test_mode=False)
    df = imputer.fit(data)
    return df


def mice_imp(data):
    print("Performing Iterative imputation using MICE")
    # Initialize MICEData instance
    mice_data = mice.MICEData(data)

    # Prepare MICE model formulas dynamically
    cols_with_missing = data.columns[data.isnull().any()].tolist()

    # Create a formula and perform MICE only for columns with missing data
    for column in cols_with_missing:
        other_columns = list(data.columns.drop(column))  # All columns except the current one
        formula = f"{column} ~ " + " + ".join(other_columns)
        mi_model = mice.MICE(formula, sm.OLS, mice_data)
        _ = mi_model.fit(10, 10)  # Using 10 imputations with 10 iterations each

    # Retrieve imputed data
    imputed_data = mice_data.data

    return imputed_data
