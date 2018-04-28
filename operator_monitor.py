#!/usr/bin/env python

""" Operator Monitor Program.

Program for monitoring an operator in real time for execution anomalies.
The program creates a Data Provider Object that gets ganglia metrics data from Gmond Probe and
redistributes them to the interested threads. Each thread monitors a different machine
running an operator and has a dedicated set of models and errors for evaluation. In the
main function of the program a TCP/IP socket connection is established with each machine
running the "operator_exec.py" module and the execution is started together with the anomaly
detection thread.

"""
import time
import sys
import Queue
import socket
import ConfigParser
import threading
import model_trainer
import data_source_factory
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

class Plotter(threading.Thread):
    """ Threaded class for plotting detected anomalies. AnomalyDetector threads
    push data to the queue and then plots are being created. This functionality must be
    implemented as an intepended thread since matplotlib uses tkinter which is restricted
    to being used only in the main thread and none else. Workarounds are almost impossible.

    Attributes:
        queue (obj): Queue where events are being pushed.
        running (bool): If false thread self-terminates.

    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue()
        self.running = True

    def run(self):
        while self.running is True:
            plt_request = self.queue.get()
            self.create_2d_plot(plt_request[0], plt_request[1], plt_request[2], \
                plt_request[3], plt_request[4], plt_request[5])
            time.sleep(1)

    def create_2d_plot(self, real, pred, warn1, warn2, machine, metric):
        """ Plots real and predicted metrics. Also shows warnings as vertical lines placed
        at the time the warning was issued.

        Args:
            real (array-like): Real metric data.
            pred (array-like): Predicted data.
            warn1 (int): Time of first warning.
            warn2 (int): Time of second warning.
            machine (str): Name of the machine being monitored.
            metric (str): Name of the metric being plotted.

        """
        # Plot graphs
        plt.plot(real)
        plt.plot(pred)

        # Plot warnings
        plt.axvline(warn1, color='#ff9900')
        plt.axvline(warn2, color='r')

        # Add colored legend.
        blue_patch = mpatches.Patch(color='blue', label='Real data')
        green_patch = mpatches.Patch(color='green', label='Predicted data')
        orange_patch = mpatches.Patch(color='orange', label='First Warning')
        red_patch = mpatches.Patch(color='red', label='Second Warning')
        plt.legend(handles=[blue_patch, green_patch, orange_patch, red_patch])

        # Save as png image.
        plt.savefig("/home/yandall/Desktop/" + machine + "-" + metric + ".png")

        #Show plot in window.
        #plt.show()

        # Clear plot.
        plt.clf()

    def kill(self):
        """ Terminate the thread. """
        self.running = False

class MetricsBroker(threading.Thread):
    """ Metrics Broker maintains a list with interested thread objects and provides them with
    data from the Gmond Probe class.

    Attributes:
        interested_threads (list): Threads to send data to.
        running (bool): If false MetricsBroker terminates.

    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.metrics = (self.cfgparser.get("Gmond", "Metrics")).split(",")
        self.interested_threads = []
        self.running = True

    def assign_interested_thread(self, thread):
        """ Add a thread to the interested list so that MetricsBroker can
        send ganglia metrics.

        Args:
            thread (obj): Thread to be added.
        """
        self.interested_threads.append(thread)
        print "append"

    def remove_interested_thread(self, thread):
        """ Remove a thread from the interested list.

        Args:
            thread (obj): Thread to be removed.
        """
        for item in self.interested_threads:
            if item.machine == thread.machine:
                self.interested_threads.remove(item)
                break

    def run(self):
        # Get data sources.
        data_sources = []
        dsf = data_source_factory.DataSourceFactory()
        for metric in self.metrics:
            data_sources.append(dsf.get_source(metric))
        for source in data_sources:
            source.start()
        # Start collecting data.
        while self.running is True:
            for source in data_sources:
                data = source.queue.get()
                if self.interested_threads:
                    print "Ok"
                    for thread in self.interested_threads:
                        try:
                            #if thread.machine == data[0]:
                            thread.queue.put(data[1])
                            #   break
                        except Queue.Full:
                            print "Unable to access queue: "
                        except ValueError:
                            print "Unable to access queue: "
                        except ReferenceError:
                            print "Unable to access queue: "
        for source in data_sources:
            source.kill()

    def kill(self):
        """ Terminate the thread. """
        self.running = False


class AnomalyDetector(threading.Thread):
    """ Anomaly detector class.

    Creates a gmond probe object, reads the data provided and performs checks to
    detect execution anomalies.

    Attributes
        cfgparser (obj): Configuration file parser.
        granularity (int): Determines how fine grained the train/test set will be (time scale).
        metrics (str list): Metrics that will be modelled / monitored.
        min_elapsed (int): Minimum execution time in seconds to allow to pass
                            before minotring begins.
        moving_avg_window_size (int): Size of the moving average window.
        warning_threshold (int): Amount of consecutive errors detected to issue a warning.
        check_interval (int): Time in seconds between two checks.
        queue (obj): Thread communication queue.
        running (bool): If false the AnomalyDetector terminates.
        machine (str): Name of the machine being monitored.
        input_size (int): Size of the operator's input in bytes.
        parameters (int list): Operator's execution parameters.
        models (dictionary): Dictionary containing the models, errors and input scalers
                                for each metric.
    """

    def __init__(self, machine, input_size, parameters, models, plotter):
        threading.Thread.__init__(self)
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.granularity = int(self.cfgparser.get("Model", "Granularity"))
        self.metrics = (self.cfgparser.get("Gmond", "Metrics")).split(",")
        self.min_elapsed = int(self.cfgparser.get("Detection", "Min_Time_Elapsed"))
        self.moving_avg_window_size = int(self.cfgparser.get("Detection", "MA_Window"))
        self.warning_threshold = int(self.cfgparser.get("Detection", "Warning_threshold"))
        self.check_interval = int(self.cfgparser.get("Detection", "Check_interval"))
        self.queue = Queue.Queue()
        self.running = True
        self.machine = machine
        self.input_size = int(input_size)
        self.parameters = map(int, parameters.split('_'))
        self.models = models
        self.plotter = plotter

    def run(self):
        # Create predictions.
        x_dur = []
        # Double assign to prevent single-element array incompatibility.
        x_dur.append([self.input_size] + self.parameters)
        x_dur.append([self.input_size] + self.parameters)
        x_dur_std = (self.models["duration"]["scaler"]).transform(x_dur)
        expected_duration = int(((self.models["duration"]["model"]).predict(x_dur_std))[0])
        duration_error = self.models["duration"]["error"]
        expected_duration = int(round(expected_duration + duration_error))
        print expected_duration
        expected_values = {}
        for metric in self.metrics:
            prediction = self.produce_behaviour(metric)
            expected_values[metric] = prediction

        print "test"
        # Initializations.
        elapsed = 0
        total_checks = 0
        real_acc = {}
        pred_acc = {}
        warn1 = 0
        warn2 = 0
        saved_plot = False
        # Create warning counters and flags.
        warning = {}
        for metric in self.metrics:
            warning[metric] = 0
        # Initialize data accumulators as dictionaries with metric being the key.
        for metric in self.metrics:
            real_acc[metric] = []
            pred_acc[metric] = []

        # Check behaviour for anomalies.
        while self.running is True:
            data = self.queue.get()
            print "get"
            elapsed = elapsed + 1

            # If the requirements are met start checking for anomalies.
            if (elapsed > self.min_elapsed) and (elapsed % self.check_interval == 0):
                total_checks += 1
                for metric in self.metrics:
                    # Get real data and predicted data.
                    real = float(data[self.metrics.index(metric)])
                    if metric == "load_one":
                        real = real * 100
                    real_acc[metric].append(real)
                    pred = self.get_moving_avg(min( \
                            int(round(100 * elapsed / expected_duration)), 100), \
                            expected_values[metric])
                    pred_acc[metric].append(pred)
                    sys.stdout.write("[" + self.machine + "] Real: " + str(real) + ", Predicted: " + str(pred) + ". \n")
                    sys.stdout.flush()

                    # Compare the absolute differece between real and predicted values and
                    # issue warnings if necessary.
                    if abs(real - pred) > self.models[metric]["error"]:
                        warning[metric] += 1
                    else:
                        warning[metric] = 0

                    # __________________WARNING 2_______________________________________
                    # Second type of warning is issued if the problem persists.
                    # Then plot the metric that caused the problem.
                    if warning[metric] > 2 * self.warning_threshold:
                        sys.stdout.write(Bcolors.FAIL + "[" + self.machine + "][FAILURE] " + metric + \
                            " metric constantly out of" + " normal levels. \n" + Bcolors.ENDC)
                        sys.stdout.flush()
                        if warn2 == 0:
                            warn2 = total_checks - 1
                        if not saved_plot:
                            for metr in self.metrics:
                                self.plotter.queue.put([real_acc[metr], pred_acc[metr], warn1,\
                                    warn2, self.machine, metr])
                            saved_plot = True

                    # __________________WARNING 1_______________________________________
                    # First type of warning is issued after real values diverge from predicted
                    # ones for more than the warning_threshold in seconds.
                    elif warning[metric] > self.warning_threshold:
                        sys.stdout.write(Bcolors.WARNING + "[" + self.machine + \
                            "][WARNING] Possible operator issue according to " \
                            + metric + " metrics.\n" + Bcolors.ENDC)
                        sys.stdout.flush()
                        if warn1 == 0:
                            warn1 = total_checks - 1

            # __________________WARNING 3_______________________________________
            # Check if duration is normal.
            if elapsed > (expected_duration + duration_error / 2):
                sys.stdout.write(Bcolors.FAIL + "[" + self.machine + \
                    "][WARNING] Duration exceeding prediction.\n" + Bcolors.ENDC)
                sys.stdout.flush()
                if not saved_plot:
                    for metr in self.metrics:
                        self.plotter.queue.put([real_acc[metr], pred_acc[metr], warn1,\
                            warn2, self.machine, metr])
                    saved_plot = True


    def get_moving_avg(self, percetage, expected_values):
        """ Get the average predicted value within the specified window.

        Args:
            percetage (int): The current time that has passed.
            expected_values (list): The predicted behaviour.

        Returns:
            (float): The average predicted value in the window.

        """

        half_window = int((self.moving_avg_window_size - 1) / 2)
        current = percetage - half_window
        count = 1
        avg_sum = expected_values[current - 1]
        while current < (len(expected_values) - 1) and count < self.moving_avg_window_size:
            count = count + 1
            current = current + 1
            avg_sum = avg_sum + expected_values[current - 1]
        return round(avg_sum / count, 1)

    def produce_behaviour(self, metric):
        """ Produces the complete operator behaviour prediction as an array. The length
        of the array is the predicted duration.

        Args:
            metric (str): The metric for which the behaviour is produced.

        Returns:
            (array-like) The array with the predicted values.

        """

        x_arr = []
        for i in  range(1, self.granularity + 1):
            x_arr.append([self.input_size] + self.parameters + [i])
        x_arr_std = (self.models[metric]["scaler"]).transform(x_arr)
        y_pred = (self.models[metric]["model"]).predict(x_arr_std)
        return y_pred

    def kill(self):
        """ Terminate the thread. """
        self.running = False


class MasterThread(threading.Thread):
    """ Machine connection master thread. This class runs a thread that connects the monitoring
    node to the monitored machine via a TCP/IP socket. We use a thread to run independently many
    operators to many machines.

    Attributes:
        parameters (list): String list containing all the operator execution parameters.
        metrics_broker (obj): Class that provides data to AnomalyDetector.

    """

    def __init__(self, parameters, metrics_broker, plotter):
        threading.Thread.__init__(self)
        self.parameters = parameters
        self.metrics_broker = metrics_broker
        self.plotter = plotter

    def run(self):
        # Initialization.
        parts = self.parameters.split(" ")
        host = parts[0]
        port = int(parts[1])
        machine = parts[2]
        operator_trn = parts[3]
        operator_tst = parts[4]
        input_size = parts[5]
        parameters = parts[6]

        # Make terminal command.
        command = ""
        for i in range(7, len(parts)):
            command = command + parts[i] + " "
        command = command[:-1]
        sys.stdout.write("Command: " + command + "\n")
        sys.stdout.flush()

        # Train models for specific operator.
        trainer = model_trainer.ModelTrainer(machine, operator_trn, operator_tst)
        models = trainer.create_models()

        # Create execution anomaly detection thread.
        anomaly_detector = AnomalyDetector(machine, input_size, parameters, models, self.plotter)

        # Connect to monitored machine.
        sys.stdout.write("[Communication] Attempting to connect to " + host + \
            " at port " + str(port) + ".\n")
        sys.stdout.flush()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        sys.stdout.write("[Communication] Sending command...")
        sys.stdout.flush()
        sent = sock.send(command)
        if sent == 0:
            sys.stdout.write(Bcolors.FAIL + "Socket connection broken.\n" + Bcolors.ENDC)
            sys.stdout.flush()
        else:
            while 1:
                msg = sock.recv(2048)
                # Operator environment sends an "EXE_RDY" message when the operator
                # is ready to execute.
                if msg == "EXE_RDY":
                    sys.stdout.write("Done!\n")
                    sys.stdout.flush()
                    sent = sock.send("EXE_ACK")
                    if sent == 0:
                        sys.stdout.write(Bcolors.FAIL + "Socket connection broken.\n" + \
                            Bcolors.ENDC)
                        sys.stdout.flush()
                        break
                    # Start anomaly detection.
                    sys.stdout.write("Starting operator monitoring.\n")
                    sys.stdout.flush()
                    anomaly_detector.start()
                    self.metrics_broker.assign_interested_thread(anomaly_detector)

                # When operator finishes execution the environment sends an "EXE_END"
                # message.
                elif msg == "EXE_END":
                    self.metrics_broker.remove_interested_thread(anomaly_detector)
                    anomaly_detector.kill()
                    time.sleep(2)
                    sys.stdout.write("Operator finished.\n")
                    sys.stdout.flush()
                    break
        # After operator have been executed, terminate the connection.
        sent = sock.send("TERMINATE")
        if sent == 0:
            sys.stdout.write(Bcolors.FAIL + "Socket connection broken.\n" + Bcolors.ENDC)
            sys.stdout.flush()
        else:
            sys.stdout.write("Terminating connection.\n")
            sys.stdout.flush()
        sock.close()


def main():
    """ Main function. """

    # Create metrics broker.
    metrics_broker = MetricsBroker()
    metrics_broker.start()
    plotter = Plotter()
    plotter.start()

    # For each line of commands spawn a dedicated thread.
    with open(sys.argv[1]) as inputfile:
        lines = inputfile.read().splitlines()
        for item in lines:
            print item
            master_thread = MasterThread(item, metrics_broker, plotter)
            master_thread.start()

if __name__ == "__main__":
    main()
