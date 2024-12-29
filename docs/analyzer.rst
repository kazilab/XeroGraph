========
Analysis
========


Perform normality test for each feature
=======================================

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
================================================

The `Kolmogorov-Smirnov (KS) test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html>`_ is a non-parametric test used to determine if a sample comes from a population with a specific distribution. It compares the empirical distribution function (EDF) of the sample with the cumulative distribution function (CDF) of the specified theoretical distribution, and it provides a test statistic that measures the largest discrepancy between them. The test is useful for comparing a sample with a reference probability distribution or comparing two samples to check if they come from the same distribution.

.. code-block:: python

    xg_test.ks()

The "ks()" function compares the sample with a normal distribution and returns two values:
    (1) statistic: This is the KS statistic, which quantifies the maximum distance between the empirical distribution function of the sample and the cumulative distribution function of the specified normal distribution. A larger KS statistic indicates a greater divergence between the two distributions.
    (2) p-value: This is the p-value, which tells you the probability of observing a test statistic at least as extreme as the one observed under the null hypothesis, which states that the sample comes from the specified distribution. A small p-value (typically less than 0.05) leads to rejection of the null hypothesis, suggesting that the data do not follow a normal distribution.

Interpretation of results
    (1) If the p-value is small (commonly less than 0.05): Reject the null hypothesis, suggesting significant evidence that the data do not follow a normal distribution.
    (2) If the p-value is large: It fails to reject the null hypothesis, indicating insufficient evidence to conclude that the data do not follow a normal distribution.


Visualize histograms for each feature
=====================================

Histograms provide a visual representation of how data points are distributed across different intervals or "bins". We applied "Freedman-Diaconis" rule to determine the bins.

.. code-block:: python

    xg_test.histograms()


Key benefits:
    (1) Detect skewness: Histograms can identify if data for a particular feature are skewed to the left or right.
    (2) Identify modality: Histograms help in recognizing if data are unimodal (one peak), bimodal (two peaks), or multimodal (multiple peaks), which can influence the selection of appropriate statistical tests or data preprocessing techniques.
    (3) Outliers: Histograms make it easier to spot outliers which appear as bars isolated from the bulk of the data. Outliers can be the result of data entry errors, measurement errors, or actual variability in data, and may significantly affect the results of statistical analyses and predictive models.
    (4) Anomalies: Unusual patterns, such as unexpected spikes in a histogram, can indicate data issues or important insights into dataset characteristics.


Visualize density plots for each feature
========================================

Density plots are smoothed, continuous versions of histograms and are useful for visualizing the underlying distribution of the data without being tied to the choice of bins.

.. code-block:: python

    xg_test.density_plots()


Key benefits:
    (1) Smooth representation: Unlike histograms, density plots provide a smooth curve representing the distribution, which can help in identifying the shape of the distribution more clearly (e.g., bimodal, normal, skewed).
    (2) Comparison of distributions: They are particularly useful when you need to compare the distribution of data across different groups or conditions within the same plot.
    (3) Handling overlap: Density plots can handle overlap better than histograms by showing peaks where data are concentrated, even if multiple groups are plotted together.


Visualize box plots for each feature
====================================

Box plots, also known as box-and-whisker plots, provide a concise and informative summary of the distribution of data across its quartiles and are particularly useful for identifying outliers, median, and data variability.

.. code-block:: python

    xg_test.box_plots()


Key benefits:
    (1) Five-Number Summary: Each box plot provides a visual representation of the minimum, first quartile (Q1), median (second quartile, Q2), third quartile (Q3), and maximum of a dataset. This five-number summary is crucial for quickly understanding the central tendency and dispersion of the data.
    (2) Detection of Outliers: Box plots make it easy to identify outliers as points that appear outside of the whiskers, which typically extend 1.5 times the interquartile range (IQR) from the quartiles. This feature is especially useful for deciding whether to exclude outliers from further analyses or for understanding the spread and tails of the distribution.


Visualize Q-Q plots for each feature
====================================

Q-Q (quantile-quantile) plot for each feature in a dataset is a highly effective method for assessing whether the distribution of the data conforms to a theoretical distribution, typically the normal distribution.

.. code-block:: python

    xg_test.qq_plots()


Key benefits:
    (1) Visual inspection of normal distribution: A Q-Q plot provides a visual means to assess the normality of data. If the data points (quantiles of the sample data) fall approximately along a straight line, the sample can be considered normally distributed. Deviations from this line indicate departures from normality.
    (2) Sensitivity to deviations: Q-Q plots are particularly sensitive to deviations in the tails of the distribution, making them superior to other techniques like histograms or box plots for detecting outliers and skewness.
    (3) Identifying Outliers: Points that deviate significantly from the reference line in a Q-Q plot can indicate potential outliers, especially those in the tails.


Visualize missing data patterns
===============================

Visualizing missing data patterns is crucial in understanding the structure and impact of missingness in your dataset. This can guide decisions regarding data cleaning, imputation strategies, and even inform about potential biases or issues in data collection processes.

.. code-block:: python

    xg_test.missing_data()


Visualize missing percentages for both features and samples
===========================================================

Visualizing missing percentages for both features (variables) and samples (observations) in a dataset can provide crucial insights into the extent and distribution of missing data. This information is essential for effective data preprocessing and ensuring robust statistical analyses.


.. code-block:: python

    xg_test.missing_percentage()


Perform Little's MCAR test
==========================

Little's MCAR (Missing Completely at Random) test is a statistical test used to analyze the mechanism of missing data in a dataset. This test helps to determine whether the missing data are indeed MCAR, meaning that the likelihood of data being missing is the same across all observations. It contrasts with other types of missing data mechanisms, such as Missing at Random (MAR) and Missing Not at Random (MNAR), where the probability of missing data depends on the observed data or unobserved data, respectively.

Key points:
    (1) MCAR: Missing Completely at Random implies that the missingness of data is independent of both observed and unobserved data. This is the strongest form of randomness in the context of missing data.
    (2) Statistical Test: Little's MCAR test uses a chi-square test to compare observed data patterns with expected patterns if the data were MCAR. The null hypothesis (H0) is that the data are MCAR.
    (3) Outcome: The test provides a p-value: If the p-value is small (typically <0.05), it suggests that there is less than a 5% probability that the data are MCAR given the observed data patterns, leading to rejection of the null hypothesis. If the p-value is large, it suggests insufficient evidence to reject the null hypothesis, indicating that the missing data may indeed be MCAR.


.. code-block:: python

    xg_test.mcar()


Check missing type
==================

This function first check if missing values are MCAR. If it fails it then checks the likelyhood of Missing at Random (MAR) and Missing Not at Random (MNAR).

Key points:
    (1) MCAR and MAR vs MNAR test: Function provides a suggestion about the missing value type.
    (2) MAR and MNAR suggestions are for each columns.
    (3) Outcome: The test provides statistics with interpretation.


.. code-block:: python

    xg_test.missing_type()
