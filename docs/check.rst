==============================================
Check after imputation and perform comparisons
==============================================


Check Plausibility
==================

In this function, we compare imputed data distribution with the original data distribution. We calculate the statistics of each feature using `pandas.DataFrame.describe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html>`_, calculate statistical differences using `scipy.stats.ks_2samp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>`_, and overlay density plotes of original and imputed data.

.. code-block:: python

    xg_test.check_plausibility(imp_data_rf)


Compare with T-test and plot
============================

This function provides a standard error of the mean (SEM) using `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html>`_, P-values using `scipy.stats.ttest_ind <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_, and violin plots.

.. code-block:: python

    xg_test.compare_with_ttest_and_plot(imp_data_ii)


Visualize feature combination plots for each feature pair
=========================================================

Plots each feature pairs in a 2D-dimensional space.

.. code-block:: python

    xg_test.feature_combinations()

