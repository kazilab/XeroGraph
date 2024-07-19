# XeroGraph

## Description
XeroGraph is a Python package developed for researchers and data scientists to analyze and visualize missing data in datasets. It incorporates Little's MCAR test, among other statistical tools, to help users understand the mechanisms behind missing data. This package is particularly optimized for small to medium-sized datasets and offers extensive visualization options to elucidate data characteristics and integrity.

## Key Features
- **Little's MCAR Test**: Determines if the missing data in a dataset is missing completely at random.
- **Statistical Tests**: Perform normality checks and Kolmogorov-Smirnov tests to evaluate the distribution of data.
- **Advanced Visualization**: Generate histograms, density plots, box plots, Q-Q plots, and more to visualize data distributions and missing data patterns.
- **Missing Data Analysis**: Tools to visualize and quantify the extent and patterns of missing data within your dataset.
- **Missing Value Imputation**: Several options to perform missing value imputation.
- **Compare Missing Value Imputation Methods**: Tools to compare different imputation methods.
- **Compare Distribution of Imputed Data**: Tools to compare distribution of imputed data with original data.

## Installation

### Prerequisites
Ensure you have Python 3.9 or later installed. XeroGraph depends on the following Python libraries:
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- xgboost
- seaborn
- torch
- nimfa
- optuna
- tqdm
- ipywidgets

These dependencies will be automatically installed during XeroGraph's installation process.

### Setting Up a Virtual Environment
It is recommended to install XeroGraph within a virtual environment to manage dependencies effectively:

#### Create a virtual environment
```bash
python -m venv xeroenv
```

#### Activate the virtual environment
#### On Linux/Mac:
```bash
source xeroenv/bin/activate  
```
#### On Windows:
```bash
xeroenv\Scripts\activate
```

### Installing XeroGraph
#### You can install XeroGraph directly from PyPI using pip:
```bash
pip install XeroGraph
```

#### Alternatively, if you have access to the source code, navigate to the root directory of the source code and run:
```bash
python setup.py install
```

## Getting Started
### Quick Example
Here's a quick example to get you started with performing Little's MCAR test, visualizing the data and imputation.
We use XeroAnalyzer application provided in XeroGraph.
```bash
# XeroAnalyzer can be imported as XA, xa, xeroanalyzer, xero_analyzer or XeroAnalyzer
from XeroGraph import xa
import pandas as pd
```
#### Example data
```bash
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, None, 6, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 1, 6, 4, 5],
    'feature2': [4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6, 4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6],
    'feature3': [1, 2, 4, 3, 6, 2, 6, 6, None, 1, 5, 0, 3, 2, 1, 1, 2, 4, 3, None, 2, 6, 6, 1, 1, 5, 0, 3, 2, 1],
    'feature4': [4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, None, 3, 2, 1, 4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, 1, 3, 2, 1],
    'feature5': [4, 3, 4, 2, None, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 4, 3, 4, 2, 1, 6, 2, 4, 5, None, 7, 8, 9, 2, 4]
    
})
print(data.shape)
```
#### Initialize the XeroGraph analyzer
```bash
# Optional arguments:
# To save plot: save_plot=True, save_path='save path'
xg_test = xa(data, save_files=False, save_path="")
```
#### Perform normality test for each features
```bash
xg_test.normality()
```
#### Perform Kolmogorov-Smirnov test for each features
```bash
xg_test.ks()
```
#### Visualize histograms for each features
```bash
xg_test.histograms()
```
#### Visualize density plots for each features
```bash
xg_test.density_plots()
```
#### Visualize box plots for each features
```bash
xg_test.box_plots()
```
#### Visualize Q-Q plots for each features
```bash
xg_test.qq_plots()
```
#### Visualize missing data patterns
```bash
xg_test.missing_data()
```
#### Visualize missing percentages for both features and samples
```bash
xg_test.missing_percentage()
```
#### Perform Little's MCAR test
```bash
mcar_result = xg_test.mcar()
print(f"MCAR Test Result: {mcar_result}")
```
## Perform imputation of continuous data
Some of the following tools can be used for imputation of categorical data but we will mainly focus on continuous data.
#### Mean Imputation
```bash
imp_data_mean = xg_test.mean_imputation() 
```
#### Median Imputation
```bash
imp_data_median = xg_test.median_imputation() 
```
#### Most Frequent
```bash
imp_data_most_frequent = xg_test.most_frequent_imputation() 
```
#### KNN imputation
```bash
imp_data_knn = xg_test.knn_imputation() 
```
#### Iterative Imputation
```bash
imp_data_ii = xg_test.iterative_imputation(plot_convergence=False) # Optional: plot_convergence=True 
```
#### Imputation by Random Forest
```bash
imp_data_rf = xg_test.random_forest_imputation() 
```
#### Imputation by LASSO CV
```bash
imp_data_lc = xg_test.lasso_cv_imputation()
```
#### Imputation by XGBoost
```bash
imp_data_xb = xg_test.xgboost_imputation()
```
#### Imputation by Xputer
```bash
imp_data_xp = xg_test.xputer_imputation() 
```
#### Multiple Imputation by MICE
```bash
imp_data_mice = xg_test.mice_imp() 
```
## Check after imputation
#### Check Plausibility
```bash
xg_test.check_plausibility(imp_data_rf) 
```
#### Compare with T-test and plot
```bash
xg_test.compare_with_ttest_and_plot(imp_data_ii) 
```
#### Visualize feature combinations plots for each features
```bash
xg_test.feature_combinations()
```
## Perform a test to check which imputation method fits for your data
We use XeroCompare application provided in XeroGraph to compare different imputation methods.
For analysis, you may provide a dataset with minimum number of missing value as XeroCompare will remove rows with missing values.

#### Run with XeroAnalyzer
```bash
# MICE imputation is a slow process, if you want to include pass "run_mice=True".
summary = xg_test.compare_imputers(run_mice=False)()
print(summary) 
```

#### Run independently as XeroCompare
```bash
# XeroCompare can be imported as XC, xc, xerocompare, xero_compare or XeroCompare
from XeroGraph import xc
# MICE imputation is a slow process, if you want to include pass "run_mice=True".
compare_imp = xc(data, run_mice=False) 
summary = compare_imp.compare()
print(summary) 
```

## Documentation
For more detailed information on all the features and usage instructions, refer to the full documentation available at ReadTheDoc(https://xerograph.readthedocs.io).

## Contributing
Contributions to XeroGraph are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to make a contribution, including bug fixes, adding new features, and improving the documentation.

## License
XeroGraph is released under the Apache License 2.0. For more details, see the LICENSE file included with the source code.

## Contact
For help and support, please open an issue in the GitHub repository or contact the development team at XeroGraph@kazilab.se.
