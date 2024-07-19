===============
Getting Started
===============

How to use
==========

Here's a quick example to get you started with performing Little's MCAR test, visualizing the data, and imputation. We use the XeroAnalyzer application provided in XeroGraph.

Importing XeroAnalyzer
-----------------------

.. code-block:: python

    # XeroAnalyzer can be imported as XA, xa, xeroanalyzer, xero_analyzer, or XeroAnalyzer
    from XeroGraph import xa
    import pandas as pd

Example data
------------

.. code-block:: python

    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, None, 6, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 1, 6, 4, 5],
        'feature2': [4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6, 4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6],
        'feature3': [1, 2, 4, 3, 6, 2, 6, 6, None, 1, 5, 0, 3, 2, 1, 1, 2, 4, 3, None, 2, 6, 6, 1, 1, 5, 0, 3, 2, 1],
        'feature4': [4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, None, 3, 2, 1, 4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, 1, 3, 2, 1],
        'feature5': [4, 3, 4, 2, None, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 4, 3, 4, 2, 1, 6, 2, 4, 5, None, 7, 8, 9, 2, 4]
    })
    print(data.shape)

Initialize the XeroGraph analyzer
---------------------------------

.. code-block:: python

    # Optional arguments:
    # To save plot: save_plot=True, save_path='save path'
    xg_test = xa(data, save_files=False, save_path="")

Perform normality test for each feature
---------------------------------------
We applied the `Shapiro-Wilk test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html>`_ in the "normality()" function. The Shapiro-Wilk test is a statistical test used to assess the normality of a dataset. It is implemented in Python's "scipy.stats" module through the "shapiro# function.

.. code-block:: python

    xg_test.normality()

The "normality()" function returns two values:
    (1) W statistic: This is the test statistic for the Shapiro-Wilk test. It quantifies how much the sample distribution deviates from a normal     distribution.
    (2) p-value: This value helps determine whether to reject the null hypothesis. The null hypothesis for the Shapiro-Wilk test states that the data are drawn from a normal distribution.

Interpreting the "normality()" test:
    (1) W Statistic: A value close to 1 indicates that the data are more likely to be normally distributed. Values significantly lower than 1 suggest deviations from normality.
    (2) P-value: If the p-value is less than the chosen alpha level (commonly set at 0.05), then you reject the null hypothesis, suggesting that the data do not come from a normal distribution. If the p-value is greater than or equal to the alpha level, you fail to reject the null hypothesis, indicating no evidence to suggest the data are not from a normal distribution.

Practical Considerations
    (1) Sample Size: The Shapiro-Wilk test is considered reliable for sample sizes less than 2,000. For very large datasets, the test might be too sensitive, detecting small deviations from normality that are not practically significant.
    (2) Usage: While the Shapiro-Wilk test provides a formal statistical assessment of normality, it's often recommended to also look at graphical assessments like QQ-plots or histograms to visually assess the distribution.


Perform the Kolmogorov-Smirnov test for each feature
------------------------------------------------
The `Kolmogorov-Smirnov (KS) test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html>`_ is a non-parametric test used to determine if a sample comes from a population with a specific distribution. It compares the empirical distribution function (EDF) of the sample with the cumulative distribution function (CDF) of the specified theoretical distribution, and it provides a test statistic that measures the largest discrepancy between them. The test is useful for comparing a sample with a reference probability distribution or comparing two samples to check if they come from the same distribution.

.. code-block:: python

    xg_test.ks()

The "ks()" compares the sample with a normal distribution and returns two values:
    (1) statistic: This is the KS statistic, which quantifies the maximum distance between the empirical distribution function of the sample and the cumulative distribution function of the specified normal distribution. A larger KS statistic indicates a greater divergence between the two distributions.
    (2) p-value: This is the p-value, which tells you the probability of observing a test statistic at least as extreme as the one observed under the null hypothesis, which states that the sample comes from the specified distribution. A small p-value (typically less than 0.05) leads to rejection of the null hypothesis, suggesting that the data do not follow a normal distribution.

Interpretation of "ks()" test results
    (1) If the p-value is small (commonly less than 0.05): Reject the null hypothesis, suggesting significant evidence that the data do not follow a normal distribution.
    (2) If the p-value is large: It fails to reject the null hypothesis, indicating insufficient evidence to conclude that the data do not follow a normal distribution.


Visualize histograms for each feature
-------------------------------------
Histograms provide a visual representation of how data points are distributed across different intervals or "bins". We applied "Freedman-Diaconis" rule to determine the bins. Visualizing histograms for each feature can provide various information including:
    (1) Detect skewness: Histograms can identify if data for a particular feature are skewed to the left or right.
    (2) Identify modality: Histograms help in recognizing if data are unimodal (one peak), bimodal (two peaks), or multimodal (multiple peaks), which can influence the selection of appropriate statistical tests or data preprocessing techniques.
    (3) Outliers: Histograms make it easier to spot outliers which appear as bars isolated from the bulk of the data. Outliers can be the result of data entry errors, measurement errors, or actual variability in data, and may significantly affect the results of statistical analyses and predictive models.
    (4) Anomalies: Unusual patterns, such as unexpected spikes in a histogram, can indicate data issues or important insights into dataset characteristics.

.. code-block:: python

    xg_test.histograms()

Visualize density plots for each feature
----------------------------------------
Density plots are smoothed, continuous versions of histograms and are useful for visualizing the underlying distribution of the data without being tied to the choice of bins.
    (1) Smooth representation: Unlike histograms, density plots provide a smooth curve representing the distribution, which can help in identifying the shape of the distribution more clearly (e.g., bimodal, normal, skewed).
    (2) Comparison of distributions: They are particularly useful when you need to compare the distribution of data across different groups or conditions within the same plot.
    (3) Handling overlap: Density plots can handle overlap better than histograms by showing peaks where data are concentrated, even if multiple groups are plotted together.

.. code-block:: python

    xg_test.density_plots()

Visualize box plots for each feature
------------------------------------
Box plots, also known as box-and-whisker plots, provide a concise and informative summary of the distribution of data across its quartiles and are particularly useful for identifying outliers, median, and data variability.

Key Benefits of Box Plots
    (1) Five-Number Summary: Each box plot provides a visual representation of the minimum, first quartile (Q1), median (second quartile, Q2), third quartile (Q3), and maximum of a dataset. This five-number summary is crucial for quickly understanding the central tendency and dispersion of the data.
    (2) Detection of Outliers: Box plots make it easy to identify outliers as points that appear outside of the whiskers, which typically extend 1.5 times the interquartile range (IQR) from the quartiles. This feature is especially useful for deciding whether to exclude outliers from further analyses or for understanding the spread and tails of the distribution.

.. code-block:: python

    xg_test.box_plots()

Visualize Q-Q plots for each feature
------------------------------------
Q-Q (quantile-quantile) plot for each feature in a dataset is a highly effective method for assessing whether the distribution of the data conforms to a theoretical distribution, typically the normal distribution.

Key Benefits of Q-Q Plots:
    (1) Visual inspection of normal distribution: A Q-Q plot provides a visual means to assess the normality of data. If the data points (quantiles of the sample data) fall approximately along a straight line, the sample can be considered normally distributed. Deviations from this line indicate departures from normality.
    (2) Sensitivity to deviations: Q-Q plots are particularly sensitive to deviations in the tails of the distribution, making them superior to other techniques like histograms or box plots for detecting outliers and skewness.
    (3) Identifying Outliers: Points that deviate significantly from the reference line in a Q-Q plot can indicate potential outliers, especially those in the tails.

.. code-block:: python

    xg_test.qq_plots()

Visualize missing data patterns
-------------------------------

.. code-block:: python

    xg_test.missing_data()

Visualize missing percentages for both features and samples
-----------------------------------------------------------

.. code-block:: python

    xg_test.missing_percentage()

Perform Little's MCAR test
--------------------------

.. code-block:: python

    mcar_result = xg_test.mcar()
    print(f"MCAR Test Result: {mcar_result}")

Imputation methods
==================

Perform imputation of continuous data

Mean Imputation
---------------

.. code-block:: python

    imp_data_mean = xg_test.mean_imputation()
    # to export data as CSV
    imp_data_mean.to_csv('mean_imputed_data.csv')

Median Imputation
-----------------

.. code-block:: python

    imp_data_median = xg_test.median_imputation()
    # to export data as CSV
    imp_data_median.to_csv('median_imputed_data.csv')

Most Frequent Imputation
------------------------

.. code-block:: python

    imp_data_most_frequent = xg_test.most_frequent_imputation()
    # to export data as CSV
    imp_data_most_frequent.to_csv('most_frequent_imputed_data.csv')

KNN Imputation
--------------

.. code-block:: python

    imp_data_knn = xg_test.knn_imputation()
    # to export data as CSV
    imp_data_knn.to_csv('KNN_imputed_data.csv')

Iterative Imputation
--------------------

.. code-block:: python

    imp_data_ii = xg_test.iterative_imputation(plot_convergence=False)
    # to export data as CSV
    imp_data_ii.to_csv('Iterative_imputed_data.csv')

Imputation by Random Forest
---------------------------

.. code-block:: python

    imp_data_rf = xg_test.random_forest_imputation()
    # to export data as CSV
    imp_data_rf.to_csv('RandomForest_imputed_data.csv')

Imputation by LASSO CV
----------------------

.. code-block:: python

    imp_data_lc = xg_test.lasso_cv_imputation()
    # to export data as CSV
    imp_data_lc.to_csv('LASSOCV_imputed_data.csv')

Imputation by XGBoost
---------------------

.. code-block:: python

    imp_data_xb = xg_test.xgboost_imputation()
    # to export data as CSV
    imp_data_xb.to_csv('XGBoost_imputed_data.csv')

Imputation by Xputer
--------------------

.. code-block:: python

    imp_data_xp = xg_test.xputer_imputation()
    # to export data as CSV
    imp_data_xp.to_csv('Xputer_imputed_data.csv')

Multiple Imputation by MICE
---------------------------

.. code-block:: python

    imp_data_mice = xg_test.mice_imp()
    # to export data as CSV
    imp_data_mice.to_csv('MICE_imputed_data.csv')

Check after imputation and perform comparisons
==============================================

Check Plausibility
------------------

.. code-block:: python

    xg_test.check_plausibility(imp_data_rf)

Compare with T-test and plot
----------------------------

.. code-block:: python

    xg_test.compare_with_ttest_and_plot(imp_data_ii)

Visualize feature combination plots for each feature pair
---------------------------------------------------------

.. code-block:: python

    xg_test.feature_combinations()

Comparison with XeroCompare
===========================

Perform a test to check which imputation method fits your data. We use the XeroCompare application provided in XeroGraph to compare different imputation methods. For analysis, you may provide a dataset with the minimum number of missing values as XeroCompare will remove rows with missing values.

.. code-block:: python

    from XeroGraph import xc
    # MICE imputation is a slow process, if you want to include pass "run_mice=True".
    compare_imp = xc(data, run_mice=False)
    summary = compare_imp.compare()
    print(summary)
