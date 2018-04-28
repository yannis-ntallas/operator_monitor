#!/usr/bin/env python

""" Operator Behaviour Models Utility Library

This library contains utility functions used by the models of the operator monitoring
program.

"""
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

def normalize_data(x_train, x_test):
    """ Create a standard scaler and fit it to the data.

    Args:
        x_train (array-like) : Data used to fit the scaler.
        data (array-like) : In case we need to use the scaler to fit other datasets
                            related to x_train.

    Returns:
        Normalized data.

    """

    std_scaler = StandardScaler()
    std_scaler.fit(x_train)
    x_train_std = std_scaler.transform(x_train)
    x_test_std = std_scaler.transform(x_test)
    return (x_train_std, x_test_std, std_scaler)


def smoothen_data(data, n_iter):
    """ Smoothen a curve with the Savitzky - Golay Filter method.

    Args:
        data (float/double array) : The input data series.

    Returns:
        (float/double array) : Smoothened data series.

    .. _Scipy Savitzky-Golay Filter:
       https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html

    """
    if len(data) > 18:
        for i in range(n_iter):
            smoothened_data = savgol_filter(data, 9, 2)
            rounded_s_data = [round(elem, 1) for elem in smoothened_data]
        return rounded_s_data
    else:
        return data
