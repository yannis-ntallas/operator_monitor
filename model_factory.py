#!/usr/bin/env python

""" Model Factory Class """

import ConfigParser
import machine_learining

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

class ModelFactory(object):
    """ Model Factory Class.
    Determines which model constructor will be called
    based on the type of metric.

    Attributes:
        cfgparser (obj): Configuration file parser.
        slp_metrics (list): Metrics modeled by a single layer perceptron.
        mlp_metrics (list): Metrics modeled by a multilayer perceptron regressor.

    """
    def __init__(self):
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.slp_metrics = (self.cfgparser.get("Model", "SLP")).split(",")
        self.mlp_metrics = (self.cfgparser.get("Model", "MLP")).split(",")

    def get_model(self, metric):
        """ Return the model for the given metric.

        Args:
            metric (str): Metric type.

        Returns:
            (obj): The model.

        """
        if metric == "duration":
            return machine_learining.DurationMultiLayerPerceptronRegressor()
        elif metric in self.slp_metrics:
            return machine_learining.SingleLayerPerceptron()
        elif metric in self.mlp_metrics:
            return machine_learining.MultiLayerPerceptronRegressor()
        else:
            print "[Model Factory] Error: No matching model!"
