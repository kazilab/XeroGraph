==================
Imputation methods
==================

Perform imputation of continuous data


Mean Imputation
===============

Mean imputation handles missing data in a dataset by replacing the missing values with the mean of the available (non-missing) values in the same variable. Implemented using `sklearn.impute.SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

Key points:
    (1) Preliminary Analysis: It might be suitable for initial exploratory data analysis when a quick, temporary fix for missing data is needed to enable broad overview analyses.
    (2) Random Missing Data: If you can reasonably assume that data are missing completely at random (MCAR), the bias introduced by mean imputation might be minimal.

.. code-block:: python

    imp_data_mean = xg_test.mean_imputation()
    # to export data as CSV
    imp_data_mean.to_csv('mean_imputed_data.csv')



Median Imputation
=================

Median imputation is a technique used to handle missing data by substituting missing values with the median of the available data for a particular variable. Implemented using `sklearn.impute.SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

Key points:
    (1) Highly Skewed Data: Median imputation is particularly useful in datasets where features are highly skewed.
    (2) Preliminary Data Analysis: It can be used in preliminary data analysis where a quick and robust method is needed to handle missing values without dropping large portions of data.
    (3) Robust Models: When the analytical methods used downstream are less sensitive to changes in variance (MCAR) but more sensitive to outliers.

.. code-block:: python

    imp_data_median = xg_test.median_imputation()
    # to export data as CSV
    imp_data_median.to_csv('median_imputed_data.csv')


Most Frequent Imputation
========================

Most Frequent Imputation, also known as Mode Imputation, involves substituting missing values with the most frequently occurring value in a dataset. While typically used for categorical data, it can also be applied to continuous data, particularly when there are repeated or common values that dominate a dataset. However, its applicability and effectiveness for continuous data are generally more limited and need careful consideration. Implemented using `sklearn.impute.SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

.. code-block:: python

    imp_data_most_frequent = xg_test.most_frequent_imputation()
    # to export data as CSV
    imp_data_most_frequent.to_csv('most_frequent_imputed_data.csv')


KNN Imputation
==============

K-Nearest Neighbors (KNN) imputation is suitable for continuous data where relationships among features can help predict missing values. The "knn_imputation()" function applied the `KNNImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_ from the sklearn.impute module in Scikit-learn that utilizes the K-Nearest Neighbors approach to replace missing values using the mean of the nearest neighbors found in the training set.

Key points:
    (1) Utilizes Correlations: Unlike simpler methods like mean or median imputation, KNN imputation can exploit the underlying relationships between features to make more accurate imputations.
    (2) Flexibility: It is inherently flexible because it does not assume a specific distribution of the data and can adapt to the particular structure of the dataset.
    (3) Non-Parametric: As a non-parametric method, it does not require fitting a model and is particularly useful in scenarios where parametric assumptions cannot be satisfied.

.. code-block:: python

    imp_data_knn = xg_test.knn_imputation()
    # to export data as CSV
    imp_data_knn.to_csv('KNN_imputed_data.csv')


Iterative Imputation
====================

The iterative_imputation() function applies the base `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_ from sklearn.impute. The IterativeImputer from Scikit-learn is an advanced imputation technique that models each feature with missing values as a function of other features in a round-robin or iterative fashion. It is a flexible imputation technique based on multivariate imputation by chained equations (MICE), a strategy that models each variable with missing values conditionally on the others through specified regression models. By default the base IterativeImputer uses a linear estimator `BayesianRidge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html>`_.

.. code-block:: python

    imp_data_ii = xg_test.iterative_imputation(plot_convergence=False)
    # to export data as CSV
    imp_data_ii.to_csv('Iterative_imputed_data.csv')


Imputation by Random Forest
===========================

The random_forest_imputation() function is an implementation of `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_ where the base estimator has been replaced by `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_. This method may be useful when data display a non-linear relationship within features.

.. code-block:: python

    imp_data_rf = xg_test.random_forest_imputation()
    # to export data as CSV
    imp_data_rf.to_csv('RandomForest_imputed_data.csv')


Imputation by LASSO CV
======================

This `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_ implementation applies LASSO model with cross-validation, `LassoCV <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html>`_.

.. code-block:: python

    imp_data_lc = xg_test.lasso_cv_imputation()
    # to export data as CSV
    imp_data_lc.to_csv('LASSOCV_imputed_data.csv')


Imputation by XGBoost
=====================

The xgboost_imputation() function is an implementation of `IterativeImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_ where the base estimator has been replaced by `XGBRegressor <https://xgboost.readthedocs.io/en/stable/parameter.html>`_. This method may be useful when data display a non-linear relationship within features.

.. code-block:: python

    imp_data_xb = xg_test.xgboost_imputation()
    # to export data as CSV
    imp_data_xb.to_csv('XGBoost_imputed_data.csv')


Imputation by Xputer
====================

The `Xputer <https://github.com/kazilab/xputer>`_ is a novel imputation tool that adeptly integrates Non-negative Matrix Factorization (NMF) with the predictive strengths of XGBoost.

.. code-block:: python

    imp_data_xp = xg_test.xputer_imputation()
    # to export data as CSV
    imp_data_xp.to_csv('Xputer_imputed_data.csv')


Multiple Imputation by MICE
===========================

This function applies statsmodels models `mice <https://www.statsmodels.org/dev/generated/statsmodels.imputation.mice.MICE.html>`_  module to data sets with missing values using the ‘multiple imputation with chained equations’ (MICE) approach.


.. code-block:: python

    imp_data_mice = xg_test.mice_imp()
    # to export data as CSV
    imp_data_mice.to_csv('MICE_imputed_data.csv')

