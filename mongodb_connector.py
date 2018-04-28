#!/usr/bin/env python

""" MongoDB connector class.

This class connects to the ganglia metrics mongodb database and contains
the functions for manipulating the documents that contain the metrics.

"""

from pymongo import MongoClient

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

class DatabaseConnector(object):
    """ Database connector class. """

    def __init__(self):
        self.client = MongoClient()
        self.database = self.client.gmdb
        self.collection = self.database.gmcol


    def insert_operator(self, operator_name, machine_name):
        """ Create new mongodb document for an operator.

        Args:
            operator_name (string): The name of the operator.
            machine_name (string): The name of the machine that produced the metrics.

        """

        doc = {
            "operator" : operator_name,
            "machine" : machine_name,
        }
        if self.collection.find_one(doc) is None:
            self.collection.insert_one(doc)
            print "Operator inserted."
        else:
            print "Operator already exists."


    def delete_operator(self, operator_name, machine_name):
        """ Delete the document of an operator.

        Args:
            operator_name (string): The name of the operator.
            machine_name (string): The name of the machine that produced the metrics.

        """

        result = self.collection.delete_one({"operator" : operator_name, "machine" : machine_name})
        del_count = result.deleted_count
        if del_count is 0:
            print "Error: Couldn't delete document. No such document."
        elif del_count is 1:
            print "Operator deleted."


    def get_metrics(self, operator_name, machine_name, metric_type):
        """ Get metrics from the mongodb document.

        Args:
            operator_name (string): The name of the operator.
            machine_name (string): The name of the machine that produced the metrics.
            metric_type (string) : Refering to the source of the metrics (CPU, Load, etc).

        Returns:
            string or list of string : The metrics corersponding to the param string.

        """

        query = self.collection.find_one({"operator" : operator_name, "machine" : machine_name})
        return query["metrics"][metric_type]
        #return query

    def insert_metrics(self, operator_name, machine_name, metric_type, input_size, \
                       parameters, time, data):
        """ Insert metrics to mongodb subdocument.

        Args:
            operator_name (string): The name of the operator.
            machine_name (string): The name of the machine that produced the metrics.
            metric_type (string) : Refering to the source of the metrics (CPU, Load, etc).
            input_size (int) : Input file's size in bytes.
            parameters (string) : The parameters of the operator execution.
            metrics (list of strings) : The metrics of the program

        """
        self.collection.update_one(
            {"operator" : operator_name, "machine" : machine_name}, {
                "$addToSet" : {
                    "metrics." + metric_type : [{"size" : input_size, \
                        "parameters" : parameters, "time" : time, "data" : data}]
                }
            }
        )

    def delete_metrics(self, operator_name, machine_name, metric_type, input_size, parameters):
        """ Remove metrics from a mongodb subdocument.

        Args:
            operator_name (string): The name of the operator.
            machine_name (string): The name of the machine that produced the metrics.
            metric_type (string) : Refering to the source of the metrics (CPU, Load, etc).
            input_size (int) : Input file's size in bytes.
            parameters (string) : The parameters of the operator execution.

        """
        self.collection.update(
            {"operator" : operator_name, "machine" : machine_name}, {
                "$pull" : {
                    "metrics." + metric_type : {"size" : input_size, "parameters" : parameters}
                }
            }
        )
