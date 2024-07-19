====================================
Compare different imputation methods
====================================

Perform a test to check which imputation method fits your data. We use the XeroCompare application provided in XeroGraph to compare different imputation methods. For analysis, you may provide a dataset with the minimum number of missing values as XeroCompare will remove rows with missing values. The application can be implemented under XeroAnalyzer or independently as XeroCompare.

With XeroAnalyzer
=================


.. code-block:: python

    xg_test = xa(data)
    # MICE imputation is a slow process, if you want to include pass "run_mice=True".
    summary = xg_test.compare_imputers(run_mice=False)
    print(summary)


Independently as XeroCompare
===========================

.. code-block:: python

    from XeroGraph import xc
    # MICE imputation is a slow process, if you want to include pass "run_mice=True".
    compare_imp = xc(data, run_mice=False)
    summary = compare_imp.compare()
    print(summary)
