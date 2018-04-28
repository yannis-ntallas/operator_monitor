#!/usr/bin/env python

""" Evaluation utilities library.

Functions used to evaluate the accuracy of the models used by the operator monitor.

"""
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

def absolute_accuracy_score(y_real, y_pred, error):
    """ The built-in score function is too harsh. So we calculate the percentage of accurate
    (within a given error margin) results.

    Args:
        y_real (array - like) : Output test data
        y_pred (array - like) : Machine learning model predictions.
        error (int) : A fixed value error.

    Returns:
        float : The percentage of predictions that have an absolute distance less or
                equal to the error.
    """
    y_pred_fixed = [int(i) for i in y_pred]
    y_real_fixed = [int(i) for i in y_real]
    a_lst = np.asarray(y_pred_fixed)
    b_lst = np.asarray(y_real_fixed)

    errors = (abs(a_lst - b_lst) > (error)).sum()
    total = int(len(y_pred))
    err = float(errors)
    ttl = float(total)
    accuracy = (ttl - err) / ttl * 100
    return accuracy

def root_mean_squared_error(y_real, y_pred):
    """ Calculates the root of the mean squared error of the predicted data
    compared to the test values.

    Args:
        y_real (array - like) : Output test data
        y_pred (array - like) : Machine learning model predictions.

    Returns:
        double : Root mean square error.

    """
    rms = sqrt(mean_squared_error(y_pred, y_real))
    rmsr = round(rms, 2)
    return rmsr
