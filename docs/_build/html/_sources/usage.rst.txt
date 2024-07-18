=============================
Setting Up a Virtual Environment
=============================

It is recommended to install XeroGraph within a virtual environment to manage dependencies effectively:

Create a virtual environment
----------------------------

.. code-block:: bash

    python -m venv xeroenv

Activate the virtual environment
--------------------------------

On Linux/Mac:

.. code-block:: bash

    source xeroenv/bin/activate

On Windows:

.. code-block:: bash

    xeroenv\\Scripts\\activate

Installing XeroGraph
====================

You can install XeroGraph directly from PyPI using pip:

.. code-block:: bash

    pip install XeroGraph

Alternatively, if you have access to the source code, navigate to the root directory of the source code and run:

.. code-block:: bash

    python setup.py install

Getting Started
===============

Quick Example
-------------

Here's a quick example to get you started with performing Little's MCAR test, visualizing the data and imputation. We use XeroAnalyzer application provided in XeroGraph.

.. code-block:: python

    # XeroAnalyzer can be imported as XA, xa, xeroanalyzer, xero_analyzer or XeroAnalyzer
    from XeroGraph import xa
    import pandas as pd

Example data:

.. code-block:: python

    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, None, 6, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 1, 6, 4, 5],
        'feature2': [4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6, 4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6],
        'feature3': [1, 2, 4, 3, 6, 2, 6, 6, None, 1, 5, 0, 3, 2, 1, 1, 2, 4, 3, None, 2, 6, 6, 1, 1, 5, 0, 3, 2, 1],
        'feature4': [4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, None, 3, 2, 1, 4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, 1, 3, 2, 1],
        'feature5': [4, 3, 4, 2, None, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 4, 3, 4, 2, 1, 6, 2, 4, 5, None, 7, 8, 9, 2, 4]
    })
    print(data.shape)

Initialize the XeroGraph analyzer:

.. code-block:: python

    # Optional arguments:
    # To save plot: save_plot=True, save_path='save path'
    xg_test = xa(data, save_files=False, save_path="")

Perform normality test for each features:

.. code-block:: python

    xg_test.normality()

Visualize various statistical plots and perform imputation:

.. code-block:: python

    xg_test.ks()
    xg_test.histograms()
    xg_test.density_plots()
    xg_test.box_plots()
    xg_test.qq_plots()
    xg_test.missing_data()
    xg_test.missing_percentage()
    mcar_result = xg_test.mcar()
    print(f"MCAR Test Result: {mcar_result}")

Imputation methods demonstrated:

.. code-block:: python

    imp_data_mean = xg_test.mean_imputation()
    imp_data_median = xg_test.median_imputation()
    imp_data_most_frequent = xg_test.most_frequent_imputation()
    imp_data_knn = xg_test.knn_imputation()
    imp_data_ii = xg_test.iterative_imputation(plot_convergence=False)
    imp_data_rf = xg_test.random_forest_imputation()
    imp_data_lc = xg_test.lasso_cv_imputation()
    imp_data_xb = xg_test.xgboost_imputation()
    imp_data_xp = xg_test.xputer_imputation()
    imp_data_mice = xg_test.mice_imp()

Check after imputation and perform comparisons:

.. code-block:: python

    xg_test.check_plausibility(imp_data_rf)
    xg_test.compare_with_ttest_and_plot(imp_data_ii)
    xg_test.feature_combinations()

Comparison with XeroCompare:

.. code-block:: python

    from XeroGraph import xc
    # MICE imputation is a slow process, if you want to include pass "run_mice=True".
    compare_imp = xc(data, run_mice=False)
    summary = compare_imp.compare()
    print(summary)
