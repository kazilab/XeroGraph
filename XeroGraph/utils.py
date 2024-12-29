import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
import os
from typing import Callable
import pingouin as pg
from sklearn import impute

def freedman_diaconis(data):
    """
    Freedman-Diaconis rule to determine bin size
    """
    # Compute the interquartile range
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Compute the bin width using the Freedman-Diaconis rule
    n = len(data)
    bin_width = 2 * IQR / np.cbrt(n)

    # Compute the total number of bins
    data_range = np.max(data) - np.min(data)
    num_bins = np.ceil(data_range / bin_width)

    return int(num_bins)  # Return as integer


###############################################################################
# Little's Test for MCAR (Adapted for Python)
###############################################################################


def em_mle_estimation(data, max_iter=100, tol=1e-5, ridge=1e-6):
    """
    Estimate the mean and covariance matrix of a dataset with missing values
    using a basic EM algorithm under the assumption of multivariate normality.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Numerical array with possible NaNs (missing values).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold based on changes in log-likelihood.
    ridge : float
        Ridge (diagonal) term added to covariance if near-singular to stabilize inversions.

    Returns
    -------
    mu_hat : np.ndarray, shape (d,)
        Estimated mean vector.
    Sigma_hat : np.ndarray, shape (d, d)
        Estimated covariance matrix.
    """
    X = data.copy()
    n, d = X.shape

    # Initialize missing entries by column means
    col_means = np.nanmean(X, axis=0)
    for j in range(d):
        missing_idx = np.isnan(X[:, j])
        X[missing_idx, j] = col_means[j]

    # Initial estimates
    mu_hat = np.mean(X, axis=0)
    Sigma_hat = np.cov(X, rowvar=False)

    def log_likelihood(xx, mu, Sigma):
        # Evaluate log-likelihood roughly for checking convergence
        # We'll treat the "imputed" data as fully observed normal
        # This is not exactly the observed-data log-likelihood, but
        # suffices to track EM convergence progress.
        ddim = xx.shape[1]
        sign, logdet_val = slogdet(Sigma)
        if sign <= 0 or np.isinf(logdet_val):
            return -np.inf
        inv_Sigma = inv(Sigma)
        # constant term for MVN
        const = 0.5 * (ddim * np.log(2 * np.pi) + logdet_val)
        ll = 0.0
        for i in range(n):
            diff = xx[i] - mu
            ll -= 0.5 * diff @ inv_Sigma @ diff
        return ll - n * const

    old_ll = -np.inf

    for _ in range(max_iter):
        # E-step: conditionally impute missing values
        for i in range(n):
            row = data[i, :]
            missing = np.isnan(row)
            if not np.any(missing):
                continue  # no missing in this row

            observed = ~missing
            mu_obs = mu_hat[observed]
            mu_mis = mu_hat[missing]

            # Partition covariance
            Sigma_oo = Sigma_hat[np.ix_(observed, observed)]
            Sigma_mm = Sigma_hat[np.ix_(missing, missing)]
            Sigma_om = Sigma_hat[np.ix_(observed, missing)]
            Sigma_mo = Sigma_hat[np.ix_(missing, observed)]

            # Invert Sigma_oo (with a ridge if needed)
            Sigma_oo_ridge = Sigma_oo + np.eye(Sigma_oo.shape[0]) * ridge
            inv_Sigma_oo = inv(Sigma_oo_ridge)

            row_obs = row[observed]
            cond_mean = mu_mis + Sigma_mo @ inv_Sigma_oo @ (row_obs - mu_obs)

            # Update the imputed values in X
            X[i, missing] = cond_mean

        # M-step: update mu_hat, Sigma_hat
        mu_new = np.mean(X, axis=0)
        Sigma_new = np.cov(X, rowvar=False)

        # Add ridge to diagonal if near-singular
        Sigma_new += np.eye(d) * ridge

        # Check convergence
        ll_new = log_likelihood(X, mu_new, Sigma_new)
        if abs(ll_new - old_ll) < tol:
            mu_hat, Sigma_hat = mu_new, Sigma_new
            break

        mu_hat, Sigma_hat = mu_new, Sigma_new
        old_ll = ll_new

    return mu_hat, Sigma_hat


def check_normality(data: pd.DataFrame):
    """
    Check multivariate normality of the dataset using Henze-Zirkler test.
    """
    result = pg.multivariate_normality(data, alpha=0.05)
    return result.normal, result.hz, result.pval


def little_mcar_test(
                    X: pd.DataFrame,
                    max_iter=100,
                    tol=1e-5,
                    ridge=1e-6):
    """
    Perform Little's MCAR test for a DataFrame with missing values, assuming
    a multivariate normal distribution. Closely mirrors the BaylorEdPsych::LittleMCAR
    approach, relying on an EM-based MLE for mean and covariance.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with potential missing values (NaN).
    max_iter : int
        Maximum iterations for EM.
    tol : float
        Convergence threshold for the EM log-likelihood improvement.
    ridge : float
        Ridge (diagonal) term added to the covariance for numerical stability.

    Returns
    -------
    results : dict
        {
            "warning": warning,
            "normality": normality,
            "chi.square": float,
            "df": int,
            "p.value": float,
            "missing.patterns": int,
            "amount.missing": pd.DataFrame,
            "data": dict
        }

    Notes
    -----
    - The function returns a dictionary mimicking the R function's output structure.
    - "data" contains subsets (as DataFrames) for each unique missingness pattern.
    - "amount.missing" tabulates how many and what percentage of rows are missing
      in each column.
    - Degrees of freedom follow the standard formula:
        df = (sum of #observed_cols across patterns) - (# total variables).
    - If df <= 0, the p-value is set to NaN.
    """
    # Check for normality
    imputer = impute.SimpleImputer(strategy='mean')
    imputed_ = imputer.fit_transform(X)
    normality, hz_stat, pval = check_normality(imputed_)
    warning = None
    if not normality:
        warning = (
            f"Data does not follow multivariate normality "
            f"(Henze-Zirkler Test Statistic (HZ)={hz_stat:.4f}, p-value={pval:.4f}). "
            f"\n Little's MCAR test may be unreliable."
        )
    
    # Convert to numpy
    data_np = X.to_numpy(dtype=float)
    n, d = data_np.shape

    # Identify missingness matrix
    r = np.isnan(data_np).astype(int)  # 1 if missing, else 0

    # Count missing per column
    nmis = r.sum(axis=0)

    # Create pattern IDs via binary representation
    # Example: if row i has missing pattern [0,1,0], we interpret that as 0*1 + 1*2 + 0*4 = 2 -> +1 => ID=3
    # This matches the R code's (r %*% 2^((1:n.var - 1))) + 1 approach.
    powers_of_2 = 2 ** np.arange(d)
    mdp = r.dot(powers_of_2) + 1  # shape (n, )

    # Build DF with pattern ID
    df_with_pattern = pd.DataFrame(data_np.copy(), columns=X.columns)
    df_with_pattern["MisPat"] = mdp

    # Unique patterns
    unique_patterns = np.sort(df_with_pattern["MisPat"].unique())
    n_mis_pat = len(unique_patterns)

    # Estimate global mean and covariance using EM
    mu_hat, Sigma_hat = em_mle_estimation(data_np, max_iter=max_iter, tol=tol, ridge=ridge)

    # Re-map pattern IDs from 1..n_mis_pat
    pattern_map = {pat_val: i + 1 for i, pat_val in enumerate(unique_patterns)}
    df_with_pattern["MisPat"] = df_with_pattern["MisPat"].map(pattern_map)

    # Create subsets for each pattern
    datasets = {}
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = df_with_pattern[df_with_pattern["MisPat"] == pat_id].iloc[:, :d]
        datasets[pat_id] = subset_df

    # Degrees of freedom
    df_val = 0
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = datasets[pat_id]
        col_sums = subset_df.isna().sum(axis=0)
        # observed columns = those with zero missing in this pattern
        observed_count = (col_sums == 0).sum()
        df_val += observed_count
    df_val -= d

    # Little's MCAR chi-square
    d2 = 0.0
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = datasets[pat_id]
        n_pat = subset_df.shape[0]

        # Pattern's sample mean
        pat_mean = subset_df.mean(axis=0).values  # shape (d,)

        col_sums = subset_df.isna().sum(axis=0)
        obs_cols = np.where(col_sums == 0)[0]  # indexes of observed columns
        if len(obs_cols) == 0:
            continue  # if everything is missing, skip

        mean_diff = pat_mean[obs_cols] - mu_hat[obs_cols]
        # Cov submatrix
        Sigma_obs = Sigma_hat[np.ix_(obs_cols, obs_cols)]
        # Add ridge before inversion for stability
        Sigma_obs += np.eye(Sigma_obs.shape[0]) * ridge
        Sigma_obs_inv = inv(Sigma_obs)

        diff_vec = mean_diff.reshape(-1, 1)
        contrib = n_pat * (diff_vec.T @ Sigma_obs_inv @ diff_vec)
        d2 += contrib[0, 0]

    # p-value
    if df_val <= 0:
        p_val = np.nan
    else:
        p_val = 1 - chi2.cdf(d2, df_val)

    # Summaries
    amount_missing = pd.DataFrame(
        [nmis, nmis / n],
        index=["Number Missing", "Percent Missing"],
        columns=X.columns
    )

    results = {
        "warning": warning,
        "normality": normality,
        "chi_square": d2,
        "df": df_val,
        "p_value": p_val,
        "missing_patterns": n_mis_pat,
        "amount_missing": amount_missing,
        "data": datasets
    }
    return results


###############################################################################
# Logistic-Likelihood Tests (MAR vs. MNAR)
###############################################################################


def logistic_log_likelihood(model, X, y):
    """
    Compute the log-likelihood of a fitted LogisticRegression model.

    Parameters
    ----------
    model : LogisticRegression
        A fitted logistic regression model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary response vector.

    Returns
    -------
    float
        The log-likelihood of the model given the data.
    """
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll


def feature_wise_mar_mnar_test(X, Y):
    """
    Perform likelihood ratio tests for MAR vs. MNAR for each feature with missingness.

    MAR model: Missingness ~ other X's
    MNAR model: Missingness ~ other X's + Y

    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix (n x m) with NaN for missing entries.
    Y : np.ndarray
        Outcome vector (n,).

    Returns
    -------
    list of tuples
        Each tuple: (feature_name, LRT_statistic, p_value)
    """
    # Binary missingness indicators
    D_matrix = (~X.isna()).astype(int)

    results = []
    for feature in X.columns:
        D = D_matrix[feature].values

        # If fully observed or fully missing, skip
        if np.all(D == 1) or np.all(D == 0):
            continue

        # Build MAR model (missingness ~ other features)
        other_feats = [col for col in X.columns if col != feature]
        # Simple imputation of 0 for missing in other features
        X_other = X[other_feats].fillna(0).values

        # Fit logistic model for MAR
        mar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mar_model.fit(X_other, D)

        # Build MNAR model (missingness ~ other features + Y)
        X_mnar = np.column_stack([X_other, Y])
        mnar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mnar_model.fit(X_mnar, D)

        # Compute log-likelihoods
        ll_mar = logistic_log_likelihood(mar_model, X_other, D)
        ll_mnar = logistic_log_likelihood(mnar_model, X_mnar, D)

        # LRT statistic
        LRT_statistic = 2 * (ll_mnar - ll_mar)
        # Degrees of freedom = 1 (added Y)
        p_value = 1 - chi2.cdf(LRT_statistic, df=1)

        results.append((feature, LRT_statistic, p_value))
    return results


def feature_wise_mar_mnar_test_with_x_only(X):

    # Binary missingness indicators
    D_matrix = (~X.isna()).astype(int)

    results = []
    for feature in X.columns:
        D = D_matrix[feature].values

        # If fully observed or fully missing, skip
        if np.all(D == 1) or np.all(D == 0):
            continue

        # Build MAR model (missingness ~ other features)
        other_feats = [col for col in X.columns if col != feature]
        # Simple imputation of 0 for missing in other features
        X_other = X[other_feats].fillna(0).values
        feature_values = X[feature].fillna(0).values

        # Fit logistic model for MAR
        mar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mar_model.fit(X_other, D)

        # Build MNAR model (missingness ~ other features + feature_values)
        X_mnar = np.column_stack([X_other, feature_values])
        mnar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mnar_model.fit(X_mnar, D)

        # Compute log-likelihoods
        ll_mar = logistic_log_likelihood(mar_model, X_other, D)
        ll_mnar = logistic_log_likelihood(mnar_model, X_mnar, D)

        # LRT statistic
        LRT_statistic = 2 * (ll_mnar - ll_mar)
        # Degrees of freedom = 1 (added Y)
        p_value = 1 - chi2.cdf(LRT_statistic, df=1)

        results.append((feature, LRT_statistic, p_value))
    return results


###############################################################################
# Combined Workflow with Edge-Case Handling
###############################################################################


def test_missingness_patterns_ll(X, Y, alpha=0.05):
    """
    Combined function to:
      1. Test for MCAR.
      2. If MCAR is rejected, perform feature-wise MAR vs MNAR tests.

    Handles edge cases:
      - If there's only 1 or 2 total variables in X (plus Y).
      - If features are fully missing or fully observed.
      - If the user has extremely few complete cases.

    Parameters
    ----------
    X : pd.DataFrame
        The design matrix (n x m), potentially with NaNs.
    Y : np.ndarray
        Outcome or target array of shape (n,).
    alpha : float, optional
        Significance level for decisions. Default 0.05.

    Returns
    -------
    dict
        {
          "mcar": {
             "Chi Square": float,
             "Degrees of Freedom": int,
             "P-value": float,
             "Conclusion": str,
             "Warning": str
          },
          "mar_mnar": [
             (feature, LRT_statistic, p_value, decision_string), ...
          ]
        }
    """
    results = {
        "mcar": None,
        "mar_mnar": []
    }

    # 1) Little's test for MCAR (if we have more than 1 feature).
    p_value = None
    normality = False
    comment = None
    if X.shape[1] > 1:
        try:
            result_mcar = little_mcar_test(X)
            warning = result_mcar['warning']
            normality = result_mcar['normality']
            chi_square = round(result_mcar['chi_square'], 4)
            df = result_mcar['df']
            p_value = round(result_mcar['p_value'], 4)
            if not np.isnan(chi_square):
                if p_value < alpha:
                    decision = f"Reject MCAR at alpha={alpha} (p={p_value:.3e})."
                else:
                    decision = f"Fail to reject MCAR at alpha={alpha} (p={p_value:.3e})."
                results["mcar"] = {
                    "Chi Square": chi_square,
                    "Degrees of Freedom": df,
                    "P-value": p_value,
                    "Conclusion": decision,
                    "Warning": warning
                }
            else:
                results["mcar"] = {
                    "Chi Square": chi_square,
                    "Degrees of Freedom": df,
                    "P-value": p_value,
                    "Conclusion": "Insufficient complete cases or data to run MCAR test.",
                    "Warning": warning
                }
        except Exception as e:
            results["mcar"] = {
                "Chi Square": np.nan,
                "Degrees of Freedom": 0,
                "P-value": np.nan,
                "Conclusion": f"MCAR test failed due to: {e}",
                "Warning": warning
            }
    else:
        # If only 1 feature in X, we can't do the full matrix-based MCAR test.
        results["mcar"] = {
            "Chi Square": np.nan,
            "Degrees of Freedom": 0,
            "P-value": np.nan,
            "Conclusion": "Only 1 feature in X; Little's MCAR test requires at least 2 features.",
            "Warning": warning
        }

    # 2) If MCAR is rejected, proceed with MAR vs. MNAR tests
    if p_value is not None and p_value < alpha:
        mar_mnar_outcomes = feature_wise_mar_mnar_test(X, Y)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
    
    elif p_value is not None and p_value > alpha and normality is False:
        comment = "Although Little's MCAR test is positive, as normality test failed, we will continue with MAR-MNAR test!"
        mar_mnar_outcomes = feature_wise_mar_mnar_test(X, Y)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
            
    elif p_value is None and normality is False:
        mar_mnar_outcomes = feature_wise_mar_mnar_test(X, Y)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
            
    else:
        results["mar_mnar"].append((None, None, None, 'Test not performed due to MCAR.'))
    
    return results, comment


def test_missingness_patterns_ll_x_only(X, alpha=0.05):
    """
    Combined function to:
      1. Test for MCAR.
      2. If MCAR is rejected, perform feature-wise MAR vs MNAR tests.

    Handles edge cases:
      - If there's only 1 or 2 total variables in X (plus Y).
      - If features are fully missing or fully observed.
      - If the user has extremely few complete cases.

    Parameters
    ----------
    X : pd.DataFrame
        The design matrix (n x m), potentially with NaNs.
        Outcome or target array of shape (n,).
    alpha : float, optional
        Significance level for decisions. Default 0.05.

    Returns
    -------
    dict
        {
          "mcar": {
             "Chi Square": float,
             "Degrees of Freedom": int,
             "P-value": float,
             "Conclusion": str,
             "Warning": str
          },
          "mar_mnar": [
             (feature, LRT_statistic, p_value, decision_string), ...
          ]
        }
    """
    results = {
        "mcar": None,
        "mar_mnar": []
    }

    # 1) Little's test for MCAR (if we have more than 1 feature).
    p_value = None
    normality = False
    comment = None
    if X.shape[1] > 1:
        try:
            result_mcar = little_mcar_test(X)
            warning = result_mcar['warning']
            normality = result_mcar['normality']
            chi_square = round(result_mcar['chi_square'], 4)
            df = result_mcar['df']
            p_value = round(result_mcar['p_value'], 4)
            if not np.isnan(chi_square):
                if p_value < alpha:
                    decision = f"Reject MCAR at alpha={alpha} (p={p_value:.3e})."
                else:
                    decision = f"Fail to reject MCAR at alpha={alpha} (p={p_value:.3e})."
                results["mcar"] = {
                    "Chi Square": chi_square,
                    "Degrees of Freedom": df,
                    "P-value": p_value,
                    "Conclusion": decision,
                    "Warning": warning
                }
            else:
                results["mcar"] = {
                    "Chi Square": chi_square,
                    "Degrees of Freedom": df,
                    "P-value": p_value,
                    "Conclusion": "Insufficient complete cases or data to run MCAR test.",
                    "Warning": warning
                }
        except Exception as e:
            results["mcar"] = {
                "Chi Square": np.nan,
                "Degrees of Freedom": 0,
                "P-value": np.nan,
                "Conclusion": f"MCAR test failed due to: {e}",
                "Warning": warning
            }
    else:
        # If only 1 feature in X, we can't do the full matrix-based MCAR test.
        results["mcar"] = {
            "Chi Square": np.nan,
            "Degrees of Freedom": 0,
            "P-value": np.nan,
            "Conclusion": "Only 1 feature in X; Little's MCAR test requires at least 2 features.",
            "Warning": warning
        }

    # 2) If MCAR is rejected, proceed with MAR vs. MNAR tests
    
    if p_value is not None and p_value < alpha:
        mar_mnar_outcomes = feature_wise_mar_mnar_test_with_x_only(X)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
    
    elif p_value is not None and p_value > alpha and normality is False:
        comment = "Although Little's MCAR test is positive, as normality test failed, we will continue with MAR-MNAR test!"
        mar_mnar_outcomes = feature_wise_mar_mnar_test_with_x_only(X)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
    
    elif p_value is None and normality is False:
        mar_mnar_outcomes = feature_wise_mar_mnar_test_with_x_only(X)
        for (feature, stat, p_val) in mar_mnar_outcomes:
            if p_val < alpha:
                dec = "Likely MNAR (reject MAR)."
            else:
                dec = "No strong evidence against MAR."
            results["mar_mnar"].append((feature, stat, p_val, dec))
            
    else:
        results["mar_mnar"].append((None, None, None, 'Test not performed due to MCAR.'))

    return results, comment


# Dataset loader


def load_data(file_name: str, ext: str) -> pd.DataFrame:
    """
    Load a dataset from the default location.

    Parameters:
        file_name (str): Name of the file without extension.
        ext (str): Extension of the file (e.g., '.csv', '.tsv', '.xlsx').

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the provided file extension is not supported.
    """
    # Normalize the extension (e.g., handle 'csv' or '.csv')
    ext = f".{ext.lstrip('.')}".lower()

    # Construct the full data path
    data_path = os.path.join(os.path.dirname(__file__), 'data', f"{file_name}{ext}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load the data based on the file extension
    if ext == '.csv':
        df = pd.read_csv(data_path, index_col=0)
    elif ext in {'.tsv', '.txt'}:
        df = pd.read_csv(data_path, sep='\t', index_col=0)
    elif ext == '.xlsx':
        df = pd.read_excel(data_path, index_col=0)
    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. Supported extensions are: '.csv', '.tsv', '.txt', '.xlsx'.")

    return df


# Dictionary to register dataset loaders
DATASETS = {
    'ACTG175': ('ACTG175', '.txt'),
    'AML': ('AML', '.csv'),
    'Breast_cancer': ('Breast_cancer', '.csv'),
    'PIMA_Indian': ('PIMA_Indian', '.csv'),
    'Student_dropout': ('Student_dropout', '.csv'),
    'Simulated_data_mcar': ('data_mcar', '.csv'),
    'Simulated_data_mar': ('data_mar', '.csv'),
    'Simulated_data_mnar': ('data_mnar', '.csv')
}


def get_dataset_loader(dataset_name: str) -> Callable[[], pd.DataFrame]:
    """
    Retrieve a dataset loader function by dataset name.

    Parameters:
        dataset_name (str): Name of the dataset.

    Returns:
        Callable[[], pd.DataFrame]: A function that loads the specified dataset.

    Raises:
        KeyError: If the dataset name is not registered in DATASETS.
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Dataset '{dataset_name}' is not registered. Available datasets: {list(DATASETS.keys())}")

    file_name, ext = DATASETS[dataset_name]
    return lambda: load_data(file_name, ext)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load a dataset by name.

    Parameters:
        dataset_name (str): Name of the dataset.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    loader = get_dataset_loader(dataset_name)
    return loader()
