import math
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, probplot, shapiro, kstest, ks_2samp, ttest_ind, sem
from itertools import combinations
from sklearn.experimental import enable_iterative_imputer
from sklearn.exceptions import ConvergenceWarning
from sklearn import impute
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from statsmodels.imputation import mice
import statsmodels.api as sm
from .xputer_main import Xpute
from .utils import freedman_diaconis
from .compare import XeroCompare
# Ignore ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class XeroAnalyzer:
    """
    A class to analyze missing data patterns and apply Little's MCAR test.

    Attributes:
    data (DataFrame): A pandas DataFrame that stores the data for analysis.
    missing (DataFrame): A binary matrix indicating missing data within `data`.
    """

    def __init__(self, data, save_files=False, save_path=""):
        """
        Initialize the DataAnalyzer with data.

        Parameters:
        data (DataFrame): A pandas DataFrame with potential missing values.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data must be a pandas DataFrame.")
        
        missing_value_indicators = ['NAN', 'Nan', 'NA#', '#VALUE!', '#DIV/0!',
                                    '-', '_', '--', '---', '__', '___', '#N/A',
                                    '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
                                    '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>',
                                    'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a',
                                    'nan', 'null', '#NUM!']

        # Replace all missing value indicators with np.nan, update the df inplace
        data = data.replace(missing_value_indicators, np.nan)
        self.data = data
        self.missing = None
        self.save_files = save_files
        self.save_path = save_path
        self.index = data.index
        self.columns = data.columns
        self._preprocess_data()
        self.min_value = np.min(np.min(data, axis=0))
        self.max_value = np.max(np.max(data, axis=0))

    def _preprocess_data(self):
        """Private method to preprocess and check data integrity."""
        # Checks for numeric data types.
        if self.data.select_dtypes(include=[np.number]).empty:
            raise ValueError("Data contains non-numeric values, which are not supported.")

        # Calculates a binary matrix for missing data.
        self.missing = self.data.isnull().astype(int)

    def mcar(self):
        """
        Perform Little's MCAR test.

        Returns:
        dict: A dictionary containing the chi-square statistic, degrees of freedom, and p-value.
        """
        if self.missing.sum().sum() == 0:
            return {'chi_square_stat': None, 'df': None, 'p_value': 1.0}

        # Count occurrences of each pattern
        pattern_counts = self.missing.groupby(list(self.missing.columns)).size()

        # Calculate probabilities of each column being missing
        col_probs = self.missing.mean()

        # Calculate expected counts for each pattern
        expected_counts = []
        for index, row in pattern_counts.index.to_frame().iterrows():
            p = np.prod([col_probs[col] if row[col] == 1 else 1 - col_probs[col] for col in self.missing.columns])
            expected_counts.append(len(self.data) * p)

        # Calculate the chi-square statistic
        chi_square_stat = ((pattern_counts - expected_counts) ** 2 / expected_counts).sum()

        # Degrees of freedom: (number of patterns - 1)
        df = len(pattern_counts) - 1

        # Calculate p-value from chi-square and df
        p_value = chi2.sf(chi_square_stat, df)

        # Determine if data is MCAR based on the p-value
        if p_value > 0.05:
            print("The data is likely Missing Completely at Random (MCAR).")
        else:
            print("The data is not Missing Completely at Random (MCAR).")

        return {'chi_square_stat': chi_square_stat, 'df': df, 'p_value': p_value}

    def missing_data(self):
        """
        Visualize missing data in the DataFrame using a matrix plot and text summary of missing data counts.
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(self.missing, aspect='auto', interpolation='nearest', cmap='viridis')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.title('Matrix of Missing Data')
        plt.colorbar(label='Missing data (1: missing, 0: present)')
        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_missing_data_plot_{timestamp}.pdf"
            plt.savefig(filename)
        plt.show()

        # print("Counts of missing data per column:")
        # print(self.data.isnull().sum())

    def missing_percentage(self):
        """
        Plot the percentage of missing values for each sample and each feature using bar plots.
        """
        # Calculating percentage of missing values by column
        column_percentages = self.data.isnull().mean() * 100

        # Calculating percentage of missing values by row
        row_percentages = self.data.isnull().mean(axis=1) * 100

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))

        # Plot for each feature
        column_percentages.plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Percentage of Missing Values per Feature')
        axes[0].set_ylabel('Percentage')

        # Plot for each sample
        row_percentages.plot(kind='bar', ax=axes[1], color='orange')
        axes[1].set_title('Percentage of Missing Values per Sample')
        axes[1].set_ylabel('Percentage')

        plt.tight_layout()
        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_missing_percentage_plot_{timestamp}.pdf"
            plt.savefig(filename)
        plt.show()

    def histograms(self):
        # Define the number of rows and columns for the subplots
        features = self.data.shape[1]

        n_cols = math.ceil(features**0.5)
        n_rows = math.ceil(features/n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows*3, n_cols*3))
        fig.suptitle('Histograms for Multiple Columns')

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Loop through all columns and create a histogram in each subplot
        for i, col in enumerate(self.data.columns):
            bins = freedman_diaconis(self.data[col].dropna())
            if bins < 30:
                bins = 30
            axes[i].hist(self.data[col].dropna(), bins=bins, color='blue', alpha=0.5)
            axes[i].set_title(col)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

        # Hide any unused axes if any
        for ax in axes[len(self.data.columns):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_histograms_{timestamp}.pdf"
            plt.savefig(filename)
        plt.show()

    def density_plots(self):
        # Define the number of rows and columns for the subplots
        features = self.data.shape[1]

        n_cols = math.ceil(features ** 0.5)
        n_rows = math.ceil(features / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows*3, n_cols*3))
        fig.suptitle('Density Plots for Multiple Columns')

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            self.data[col].dropna().plot(kind='density', ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel('Value')

        for ax in axes[len(self.data.columns):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_density_plot_{timestamp}.pdf"
            plt.savefig(filename)
        plt.show()

    def box_plots(self):
        # Define the number of rows and columns for the subplots
        features = self.data.shape[1]

        n_cols = math.ceil(features ** 0.5)
        n_rows = math.ceil(features / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows*3, n_cols*3))
        fig.suptitle('Boxplot Plots for Multiple Columns')

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            # Plot the boxplot on the correct subplot axis
            self.data.boxplot(column=[col], ax=axes[i])
            # axes[i].set_title(col)
            axes[i].set_ylabel('Value')

        for ax in axes[len(self.data.columns):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_box_plot_{timestamp}.pdf"
            plt.savefig(filename)

        plt.show()

    def qq_plots(self):
        # Define the number of rows and columns for the subplots
        features = self.data.shape[1]

        n_cols = math.ceil(features ** 0.5)
        n_rows = math.ceil(features / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows*3, n_cols*3))
        fig.suptitle('Q-Q Plots for Multiple Columns')

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            # Generate a Q-Q plot for each column
            probplot(self.data[col].dropna(), dist="norm", plot=axes[i])
            axes[i].set_title(f'Q-Q plot for {col}')

        # Hide any unused axes if the number of plots is less than the number of subplots
        for ax in axes[len(self.data.columns):]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_Q_Q_plot_{timestamp}.pdf"
            plt.savefig(filename)
        plt.show()

    def normality(self):
        # Initialize a dictionary to hold the results
        normality_results = {}

        for col in self.data.columns:
            stat, p = shapiro(self.data[col].dropna())
            normality_results[col] = {'Statistics': stat, 'p-value': p}

        # Display results
        for col, result in normality_results.items():
            print(f"{col} - Statistics={result['Statistics']:.3f}, p-value={result['p-value']:.3f}")
            alpha = 0.05
            if result['p-value'] > alpha:
                print(f"{col} - Sample looks Gaussian (fail to reject H0)\n")
            else:
                print(f"{col} - Sample does not look Gaussian (reject H0)\n")

    def ks(self):
        # Kolmogorov-Smirnov test assuming normal distribution
        for col in self.data.columns:
            stat, p = kstest(self.data[col].dropna(), 'norm', args=(self.data[col].dropna().mean(),
                                                                    self.data[col].dropna().std()))
            # 'norm' indicates that the sample data is being compared to a normal (Gaussian) distribution.
            print(f'{col}: Statistics=%.3f, p=%.3f' % (stat, p))
            # Interpret results
            alpha = 0.05
            if p > alpha:
                print(f'{col}: Sample looks Gaussian (fail to reject H0)')
            else:
                print(f'{col}: Sample does not look Gaussian (reject H0)')
    
    def mean_imputation(self):
        imputer = impute.SimpleImputer(strategy='mean')
        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_mean_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df
    
    def median_imputation(self):
        imputer = impute.SimpleImputer(strategy='median')
        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_median_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df
   
    def most_frequent_imputation(self):
        imputer = impute.SimpleImputer(strategy='most_frequent')
        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_most_frequent_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df
        
    def knn_imputation(self):
        imputer = impute.KNNImputer()
        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_KNN_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df
    
    def iterative_imputation(self, plot_convergence=False, max_iter=25):
        imputer = impute.IterativeImputer(max_iter=max_iter, random_state=0,
                                          min_value=self.min_value, max_value=self.max_value,
                                          sample_posterior=plot_convergence)
        imputed = None

        if plot_convergence:
            cols_with_missing = [col for col in self.data.columns if self.data[col].isnull().any()]
            imputer.fit(self.data)
            for col in cols_with_missing:
                values = []
                for i in range(max_iter):
                    imputed = imputer.transform(self.data)
                    values.append(imputed[:, self.data.columns.get_loc(col)])

                values = np.array(values).T

                plt.figure(figsize=(10, 6))
                for value_set in values:
                    plt.plot(value_set, marker='o', linestyle='-', alpha=0.3)
                plt.title(f'Convergence check for {col}')
                plt.xlabel('Iteration number')
                plt.ylabel('Imputed values')
                plt.grid(True)
                plt.show()
        else:
            imputed = imputer.fit_transform(self.data)
        
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_IterativeImputer_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df

    def random_forest_imputation(self):
        imputer = impute.IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1),
                                          max_iter=100, random_state=0,
                                          min_value=self.min_value, max_value=self.max_value)
        imputed = imputer.fit_transform(self.data)

        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_Random_Forest_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df

    def lasso_cv_imputation(self):
        imputer = impute.IterativeImputer(estimator=LassoCV(max_iter=100000, n_jobs=-1),
                                          max_iter=1000, random_state=0,
                                          min_value=self.min_value, max_value=self.max_value)
        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_LassoCV_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df

    def xgboost_imputation(self):
        imputer = impute.IterativeImputer(estimator=XGBRegressor(n_jobs=-1), max_iter=100, random_state=0,
                                          min_value=self.min_value, max_value=self.max_value)

        imputed = imputer.fit_transform(self.data)
        df = pd.DataFrame(imputed, index=self.index, columns=self.columns)
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_XGBoost_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df

    def xputer_imputation(self):
        imputer = Xpute(impute_zeros=False, pre_imputation='MixType', xgb_models=3, mf_for_xgb=True,
                        use_transformed_df=False, optuna_for_xgb=True, optuna_n_trials=50, n_iterations=3,
                        save_imputed_df=False, save_plots=False, test_mode=False)
        df = imputer.fit(self.data)
        
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_Xputer_imputed_data_{timestamp}.csv"
            df.to_csv(filename)
        return df

    # Function to check plausibility for all columns with missing values
    def check_plausibility(self, imputed_data):
        cols_with_missing = [col for col in self.data.columns if self.data[col].isnull().any()]
        for col in cols_with_missing:
            original_stats = self.data[col].describe()
            imputed_stats = imputed_data[col].describe()

            print(f"Original {col} Statistics:\n{original_stats}")
            print(f"Imputed {col} Statistics:\n{imputed_stats}")
            # Perform Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(self.data[col].dropna(), imputed_data[col])
            print(f"Kolmogorov-Smirnov test for {col}: Statistic={ks_stat}, P-value={ks_pvalue}")

            plt.figure(figsize=(12, 6))
            sns.kdeplot(self.data[col].dropna(), label='Original', color='blue', fill=True)
            sns.kdeplot(imputed_data[col], label='Imputed', color='red', fill=True)
            plt.title(f'Distribution comparison for {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    def compare_with_ttest_and_plot(self, imputed_data):
        # Iterate over each column in the DataFrame
        cols_with_missing = [col for col in self.data.columns if self.data[col].isnull().any()]
        for column in cols_with_missing:
            if self.data[column].dtype in ['float64', 'int64']:  # Check if the column is numeric
                # Perform T-test
                # Exclude NaN values from original data for a fair comparison
                t_stat, p_value = ttest_ind(self.data[column].dropna(), imputed_data[column], equal_var=False)
                # print(f"T-test for {column}: Statistic={t_stat}, P-value={p_value}")

                # Calculate SEM for original and imputed data
                sem_original = sem(self.data[column].dropna(), nan_policy='omit')
                sem_imputed = sem(imputed_data[column], nan_policy='omit')

                # Plotting violin plots
                plt.figure(figsize=(8, 6))
                data_to_plot = pd.DataFrame({
                    'Original': self.data[column],
                    'Imputed': imputed_data[column]
                })
                sns.violinplot(data=data_to_plot)
                plt.title(f'Violin Plot of {column} \n SEM Ori: {sem_original:.3f}; SEM Imp: {sem_imputed:.3f} \n P-value={p_value}')

                # Add SEM text
                # plt.text(0, 1.05, f'SEM Ori: {sem_original:.3f}', ha='center', va='bottom',
                #          transform=plt.gca().transAxes)
                # plt.text(1, 1.05, f'SEM Imp: {sem_imputed:.3f}', ha='center', va='bottom',
                #          transform=plt.gca().transAxes)

                plt.show()

    def mice_imp(self):
        # Initialize MICEData instance
        mice_data = mice.MICEData(self.data)

        # Prepare MICE model formulas dynamically
        cols_with_missing = self.data.columns[self.data.isnull().any()].tolist()

        # Create a formula and perform MICE only for columns with missing data
        for column in cols_with_missing:
            other_columns = list(self.data.columns.drop(column))  # All columns except the current one
            formula = f"{column} ~ " + " + ".join(other_columns)
            mi_model = mice.MICE(formula, sm.OLS, mice_data)
            mi_results = mi_model.fit(10, 10)  # Using 10 imputations with 10 iterations each
            print(mi_results.summary())

        # Retrieve imputed data
        imputed_data = mice_data.data
        return imputed_data

    def feature_combinations(self):

        features = self.data.columns

        # Create a list of all combinations of the features taken two at a time
        feature_combinations = list(combinations(features, 2))

        # Determine the size of the grid. We will use the smallest square layout that fits all combinations.
        num_plots = len(feature_combinations)
        grid_size = int(np.ceil(np.sqrt(num_plots)))  # Calculate grid size needed to fit all plots

        # Set up the matplotlib figure and axes
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))  # Adjust the size as needed

        axes = axes.ravel()  # Flatten the 2D array of axes to simplify looping

        # Plot each combination in a separate subplot
        for ax, (feature1, feature2) in zip(axes, feature_combinations):

            ax.scatter(self.data[feature1], self.data[feature2], alpha=0.5)
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            # ax.legend(title='Label')
            ax.set_title(f'Scatter Plot of {feature1} vs {feature2}')

        # Turn off axes that aren't needed
        for ax in axes[len(feature_combinations):]:
            ax.axis('off')  # Hide unused subplots

        # Adjust layout to prevent overlap
        fig.tight_layout()

        # save plot as pdf
        if self.save_files:
            # Get the current date and time
            current_time = datetime.datetime.now()
            # Format the datetime object as a string suitable for filenames
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # Construct the filename with the date-time stamp
            filename = f"{self.save_path}_all_features_combinations_scatter_plot_{timestamp}.pdf"
            plt.savefig(filename)

        # Show the plot
        plt.show()

    def compare_imputers(self, run_mice=False):
        compare_imp = XeroCompare(self.data)
        # MICE imputation is a slow process, if you want to include pass "run_mice=True".
        summary = compare_imp.compare(run_mice=run_mice)
        return summary
