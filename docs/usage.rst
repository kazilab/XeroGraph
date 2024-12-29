===============
Getting Started
===============

How to use
==========

Here's a quick example to get you started with performing Little's MCAR test, visualizing the data, and imputation. We use the XeroAnalyzer application provided in XeroGraph.


Importing XeroAnalyzer
=======================

Import XeroAnalyzer from XeroGraph
----------------------------------
XeroGraph allows importing XeroAnalyzer several different names including XA, xa, xeroanalyzer, xero_analyzer, or XeroAnalyzer

.. code-block:: python

    # XeroAnalyzer can be imported as XA, xa, xeroanalyzer, xero_analyzer, or XeroAnalyzer
    from XeroGraph import xa
    import pandas as pd


Example data
============

Create a pandas DataFrame
-------------------------
We create an example dataset with missing values. Users must import data as pandas DataFrame. Data should only contain numerical values and missing values without any column with string. 

.. code-block:: python

    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, None, 6, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 1, 6, 4, 5],
        'feature2': [4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6, 4, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 3, 2, 2, 6],
        'feature3': [1, 2, 4, 3, 6, 2, 6, 6, None, 1, 5, 0, 3, 2, 1, 1, 2, 4, 3, None, 2, 6, 6, 1, 1, 5, 0, 3, 2, 1],
        'feature4': [4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, None, 3, 2, 1, 4, 3, 1, 2, 4, 5, 6, 7, 8, 9, 2, 1, 3, 2, 1],
        'feature5': [4, 3, 4, 2, None, 6, 2, 4, 5, 6, 7, 8, 9, 2, 4, 4, 3, 4, 2, 1, 6, 2, 4, 5, None, 7, 8, 9, 2, 4]
    })
    print(data.shape)


Example datasets
-------------------------
We provide severa example datasets. 

.. code-block:: python

   from XeroGraph import DATASETS, load_dataset

    # List available datasets
    print("Available datasets:", list(DATASETS.keys()))

    # Load a specific dataset
    df = load_dataset('ACTG175')



Initialize the XeroAnalyzer
===========================

Initialize the XeroAnalyzer to perform analysis
-----------------------------------------------
To perform analysis, we first need to initialize XeroAnalyzer. XeroAnalyzer requires a pandas DataFrame that includes numerical values and may contain missing values in any column with strings. Additionally, we can pass "save_plot=True" to save all plots and output data. We should also specify a save location by passing a save path, like so: "save_path='save_path'".

.. code-block:: python

    # Optional arguments:
    # To save plot: save_plot=True, save_path='save path'
    xg_test = xa(data, save_files=False, save_path="")
