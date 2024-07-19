import numpy as np

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
