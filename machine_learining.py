#!/usr/bin/env python

""" Machine learning library.

Core classes for implementing machine learning models. Each model must have 2
methods implemented:
    train(x, y): Trains the model. x is the input and y is the output.
    predict(x): Predicts the output of x input using a trained model.

"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"


###---------------------------------Value Models--------------------------------------------/

class SingleLayerPerceptron(object):
    """ Single-Layer Perceptron.

    Attributes:
        model: The trained perceptron.

    .. _MLP Regressor Documentation:
       http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

    """
    def __init__(self):
        self.model = None

    def train(self, x_train_std, y_train):
        """ Train single layer perceptron.

        Args:
            x_train_std (array-like) : Normalized training data.
            y_train (array-like) : Training data output.

        Returns:
            (obj): The trained perceptron.

        """
        y_tr = np.asarray(y_train)
        # Create and train the perceptron.
        self.model = Perceptron(n_iter=5, shuffle=True)
        self.model.fit(x_train_std, y_tr)

    def predict(self, x_pred):
        """ Use model to make a prediction.

        Args:
            x_pred (array-like): Input data.
        Return:
            (array-like): Predicted output.

        """
        return self.model.predict(x_pred)


class MultiLayerPerceptronRegressor(object):
    """ Multi-layer perceptron Regressor.

    Attributes:
        model: The trained perceptron.

    .. _MLP Regressor Documentation:
       http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    """
    def __init__(self):
        self.model = None

    def train(self, x_train_std, y_train):
        """ Create and train a multilayer perceptron regressor.

        Args:
            x_train_std (array-like) : Normalized training data.
            y_train (array-like) : Training data output.

        Returns:
            (obj): The trained perceptron.

        """
        self.model = MLPRegressor(hidden_layer_sizes=(10,), activation="logistic", \
                solver='sgd', learning_rate="adaptive", max_iter=5000, \
                shuffle=True, verbose=False)
        self.model.fit(x_train_std, y_train)

    def predict(self, x_pred):
        """ Use model to make a prediction.

        Args:
            x_pred (array-like): Input data.

        Return:
            (array-like): Predicted output.

        """
        return self.model.predict(x_pred)


###---------------------------------Duration Models--------------------------------------------/

class DurationMultiLayerPerceptronRegressor(object):
    """ Create and train a multilayer perceptron regressor specialized for execution
    duration predictions.

    Attributes:
        model: The trained perceptron.

    .. _MLP Regressor Documentation:
       http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    """
    def __init__(self):
        self.model = None

    def train(self, x_train_std, y_train):
        """ Create and train a multilayer perceptron regressor.

        Args:
            x_train_std (array-like) : Normalized training data.
            y_train (array-like) : Training data output.

        Returns:
            (obj): The trained perceptron.

        """
        self.model = MLPRegressor(hidden_layer_sizes=(300,), activation="tanh", \
                solver='sgd', learning_rate="adaptive", max_iter=5000, \
                shuffle=True, verbose=False)
        self.model.fit(x_train_std, y_train)

    def predict(self, x_pred):
        """ Use model to make a prediction.

        Args:
            x_pred (array-like): Input data.
        Return:
            (array-like): Predicted output.

        """
        return self.model.predict(x_pred)
