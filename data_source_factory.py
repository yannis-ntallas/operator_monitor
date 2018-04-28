#!/usr/bin/env python

""" Data Source Factory Class """

import ConfigParser
import gmond_probe

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3."
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

class DataSourceFactory(object):
    """ Data Source Factory Class.
    Determines which source will be used based on the type of metric.

    Attributes:
        cfgparser (obj): Configuration file parser.
        gm_metrics (list): Metrics collected using ganglia monitor daemon.

    """

    def __init__(self):
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.gm_metrics = (self.cfgparser.get("Detection", "GMMetrics")).split(",")

    def get_source(self, metric):
        """ Return the data source for the given metric.

        Args:
            metric (str): Metric type.

        Returns:
            (obj): The data source instance.

        """

        if metric in self.gm_metrics:
            return gmond_probe.GmondProbe()
        else:
            print "[Data Source Factory] Error for " + metric + " metric: No matching data sources!"
