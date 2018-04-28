#!/usr/bin/env python

""" Model Trainer Class.

Creates datasets, trains and evaluates prediction models.

"""
import sys
import ConfigParser
import model_utils
import evaluation_utils
import mongodb_connector
import model_factory

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

class Bcolors(object):
    """ Terminal ANSI colors. """
    def __init__(self):
        pass
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ModelTrainer(object):
    """ Model trainer class.

    Trains various models predicting execution behaviour.

    Attributes:
        cfgparser (obj): Configuration file parser.
        granularity (int): Determines how fine grained the train/test set will be (time scale).
        mongodb_con (obj): Ganglia metrics database connector.
        mdl_factory (obj): Model factory class instance.
        metrics (str list): Metrics that will be modelled / monitored.
        machine (str): Name of the machine being monitored.
        operator_train (str): Operator's name (trainset).
        operator_test (str): Operator's name (testset).
        n_iter (int): How many times the training process will repeat until we
                            achieve an acceptable error.
    """

    def __init__(self, machine, operator_trn, operator_tst):
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.granularity = int(self.cfgparser.get("Model", "Granularity"))
        self.mongodb_con = mongodb_connector.DatabaseConnector()
        self.mdl_factory = model_factory.ModelFactory()
        self.metrics = (self.cfgparser.get("Gmond", "Metrics")).split(",")
        self.machine = machine
        self.operator_train = operator_trn
        self.operator_test = operator_tst
        self.n_iter = int(self.cfgparser.get("Model", "Train_iterations"))


    def create_models(self):
        """ Iterates through all the metrics and creates the corresponding model

        Returns:
            A dictionary with {'metric' : {model, acc, mse, scaler}} entries.

        """
        sys.stdout.write("[Model Trainer] Starting neural network training.\n")
        sys.stdout.flush()

        models = {}
        for metric in self.metrics:
            sys.stdout.write(Bcolors.OKBLUE + "[Model Trainer] Training model for " + \
                metric + "... " + Bcolors.ENDC)
            sys.stdout.flush()
            model_dict = self.create_model_helper(self.machine, self.operator_train, \
                    self.operator_test, metric, self.n_iter)
            models[metric] = model_dict
            sys.stdout.write(Bcolors.OKBLUE + "Done!" + "\n" + Bcolors.ENDC)
            sys.stdout.flush()

        sys.stdout.write(Bcolors.OKGREEN + \
            "[Model Trainer] Training model for operator duration... " \
            + Bcolors.ENDC)
        sys.stdout.flush()
        model_dict = self.create_model_helper(self.machine, self.operator_train, \
                    self.operator_test, "duration", self.n_iter)
        models["duration"] = model_dict
        sys.stdout.write("Done!" + "\n")
        sys.stdout.flush()
        return models


    def create_model_helper(self, machine, operator_train, operator_test, metric, n_iter):
        """ Trains a behaviour prediciton model for a specific metric.
        Args:
            machine (str): Name of the machine that runs the operator.
            operator_train (str): Name of operator (train-data).
            operator_test (str): Name of operator (test-data).
            metric (str): Metric corresponding to the model.
            n_iter (int): Number of training repeats.

        Returns:
            The models and the coresponding root mean squared errors as a dictionary.

        """
        # Enable dataset dumping.
        dump = False
        if (self.cfgparser.get("Model", "Dump_dataset")).lower() == "true":
            dump = True

        # Load datasets and normalize input.
        if metric == "duration":
            trainset = self.create_duration_dataset(machine, operator_train, self.metrics[0], dump)
            testset = self.create_duration_dataset(machine, operator_test, self.metrics[0], dump)
        else:
            trainset = self.create_value_dataset(machine, operator_train, metric, dump)
            testset = self.create_value_dataset(machine, operator_test, metric, dump)
        x_train_std, x_test_std, scaler = model_utils.normalize_data(trainset[0], testset[0])

        # Load and train model.
        model = self.mdl_factory.get_model(metric)
        model.train(x_train_std, trainset[1])
        y_pred = model.predict(x_test_std)
        mse = evaluation_utils.root_mean_squared_error(y_pred, testset[1])

        acc = evaluation_utils.absolute_accuracy_score(y_pred, testset[1], \
                int(self.cfgparser.get("Accuracy", metric)))
        #self.dump(trainset[1], "train.txt")
        self.dump(testset[1], "real.txt")

        # Repeat training and evaluation n_iter times and keep the best result
        best_model = model
        best_acc = acc
        best_mse = mse
        if n_iter > 1:
            for _ in range(0, n_iter - 1):
                model = None
                model = self.mdl_factory.get_model(metric)
                model.train(x_train_std, trainset[1])
                y_pred = model.predict(x_test_std)

                acc = evaluation_utils.absolute_accuracy_score(y_pred, testset[1], \
                        int(self.cfgparser.get("Accuracy", metric)))
                mse = evaluation_utils.root_mean_squared_error(y_pred, testset[1])
                if acc > best_acc:
                    best_model = model
                    best_acc = acc
                    best_mse = mse
        self.dump(y_pred, "pred.txt")

        sys.stdout.write(Bcolors.BOLD + "Final Training Results:\n" + Bcolors.ENDC)
        sys.stdout.flush()
        sys.stdout.write("Metric:............." + metric + "\n")
        sys.stdout.flush()
        sys.stdout.write("MSE:................" + str(best_mse) + "\n")
        sys.stdout.flush()
        return {"model" : best_model, "accuracy" : best_acc, "error" : best_mse, "scaler" : scaler}


    def create_value_dataset(self, machine, operator, metric, dump):
        """ Create a granularized dataset ready for usage for training a neural network

        Args:
            operator (str): The name of the operator.
            machine (str): The name of the machine that produced the metrics.
            metric (str): Metric type of the dataset.
            dump (bool): If true program writes dataset to txt file.

        Returns:
            (list): The dataset.
        """

        x_arr = []
        y_arr = []
        # Get data from db.
        query = self.mongodb_con.get_metrics(operator, machine, metric)
        for i in range(0, len(query)):
            input_size = query[i][0]["size"]
            #print input_size
            # Parameters are a string with numbers separated using underscores. They must be split
            # Before added to the dataset.
            # Example: 15_32_1
            parameter_string = query[i][0]["parameters"]
            parameters = []
            parameters = parameter_string.split("_")
            parameters = map(int, parameters)
            tmp_data = query[i][0]["data"]
            print len(tmp_data)
            fl_data = map(float, tmp_data)
            data = model_utils.smoothen_data(fl_data, 10)
            if len(data) > 3:
                data[0] = data[3]
                data[1] = data[3]
                data[2] = data[3]

            # Data must be granularized to time percentages.
            for j in range(1, (self.granularity + 1)):
                # Get the coresponding value
                percentage = j
                index = int(round(len(data) * percentage / self.granularity))
                while not index < len(data):
                    index = index - 1
                value = data[index]

                # Add data to dataset.
                tmpx = [int(input_size)] + parameters + [int(percentage)]
                tmpy = None
                if metric == "load_one":
                    tmpy = int(value * 100)
                else:
                    tmpy = int(value)
                x_arr.append(tmpx)
                y_arr.append(tmpy)

        #Dump the dataset to a txt file.
        if dump:
            filename = machine + "_" + operator + "_" + metric + "_dataset.txt"
            txtfile = open(filename, 'a+')
            txtfile.truncate()
            for i in  range(0, len(x_arr)):
                print>>txtfile, self.array_concat(x_arr[i]) + "," + str(y_arr[i])
            txtfile.close()
        return [x_arr, y_arr]


    def create_duration_dataset(self, machine, operator, metric, dump):
        """ Create a dataset ready for usage for training a neural network

        Args:
            operator (str): The name of the operator.
            machine (str): The name of the machine that produced the metrics.
            metric (str): Metric type of the dataset.
            dump (bool): If true program writes dataset to txt file.

        Returns:
            (list): The dataset.
        """

        x_arr = []
        y_arr = []
        # Get data from db.
        query = self.mongodb_con.get_metrics(operator, machine, metric)
        for i in range(0, len(query)):
            input_size = query[i][0]["size"]
            # Parameters are a string with numbers separated using underscores. They must be split
            # Before added to the dataset.
            # Example: 15_32_1
            parameter_string = query[i][0]["parameters"]
            parameters = []
            parameters = parameter_string.split("_")
            parameters = map(int, parameters)
            duration = query[i][0]["time"]
            x_arr.append([int(input_size)] + parameters)
            y_arr.append(duration)

        # Dump the dataset to a txt file.
        if dump:
            filename = machine + "_" + operator + "_duration_dataset.txt"
            txtfile = open(filename, 'a+')
            txtfile.truncate()
            for i in  range(0, len(x_arr)):
                print>>txtfile, str(x_arr[i]) + "," + str(y_arr[i])
            txtfile.close()
        return [x_arr, y_arr]

    def dump(self, data, filename):
        """ dump """
        txtfile = open(filename, 'a+')
        txtfile.truncate()
        for i in  range(0, len(data)):
            print>>txtfile, str(data[i])
        txtfile.close()
    
    def array_concat(self, arr):
        return str(arr[0]) + "," + str(arr[1]) + "," + str(arr[2])


def main():
    """Main"""

    xr = []
    yr = []
    with open("r.txt") as testfile:
        data = testfile.read().splitlines()
        for line in data:
            lst = line.split(',')
            xr.append([int(lst[0]), int(lst[1])])
            yr.append(int(float(lst[2])))

    mdl_trn = ModelTrainer("yandall-vaio", "kmeans_train", "kmeans_test")
    #tmp = mdl_trn.create_duration_dataset("yandall-vaio", "kmeans_train", "cpu_user", 1)
    dictionary = mdl_trn.create_model_helper("localhost", "kmeans_train_small_local", \
            "kmeans_test_small_local", "cpu_user", 15)
    # print dictionary["error"]
    # xr_std = dictionary["scaler"].transform(xr)
    # yr_pred = dictionary["model"].predict(xr_std)
    # txtfile = open("pr.txt", 'a+')
    # txtfile.truncate()
    # for i in  range(0, len(yr_pred)):
    #     print>>txtfile, yr_pred[i]
    # txtfile.close()
    # txtfile = open("re.txt", 'a+')
    # txtfile.truncate()
    # for i in  range(0, len(yr)):
    #     print>>txtfile, yr[i]
    # txtfile.close()
    # print "Test mse: " + str(evaluation_utils.root_mean_squared_error(yr, yr_pred))
    # print "ACC: " + str(acc) + "%"
    # print "MSE: " + str(mse)
    # print tmp

if __name__ == "__main__":
    main()
