#!/usr/bin/env python

""" Ganglia Metrics Collector.

This module connects to the machine that is running the operator via a TCP socket,
requests star of the execution and then starts collecting data using a gmond probe.
Once the execution is finished it stores the data to the appropriate file of the local
mongodb ganglia metrics database and terminates itself.

Args:
    file (txt) containing in which every line has the parameters for one operator execution.

"""
import time
import sys
import socket
import threading
import ConfigParser
import gmond_probe
import mongodb_connector

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"


class Collector(threading.Thread):
    """ Metrics collector class.

    Creates a gmond probe and connects to the mongodb metrics database. While the thread is running
    it gathers metrics . Once termination is requested it stored the data and terminates.

    Attributes:
        running (bool): Used to check if thread is running.
        metrics (list): Contains the metric types that will be collected.
        machine (str): Name of the machine that runs the operator.
        operator (str): Name of the operator.
        input_size (int): Operator's input size in bytes.
        parameters (str): Operator's execution parameters.
        gmprobe (obj): Instance of the gmond probe class.
        mongodb_con (obj): Instance of the mongodb connector.
        smoothen (bool): If true the timeseries are smoothened using the Savitzky - Golay Filter.

    """

    def __init__(self, metrics, operator, input_size, parameters, machine):
        threading.Thread.__init__(self)
        self.running = True
        self.metrics = metrics
        self.machine = machine
        self.operator = operator
        self.input_size = input_size
        self.parameters = parameters
        self.gmprobe = gmond_probe.GmondProbe()
        self.mongodb_con = mongodb_connector.DatabaseConnector()

    def run(self):
        # Insert operator to db.
        self.mongodb_con.insert_operator(self.operator, self.machine)

        # Start collecting metrics
        collected = []
        self.gmprobe.start()
        while self.running is True:
            data = self.gmprobe.queue.get()
            if data[0] == self.machine:
                print data
                collected.append(data[1])
        self.gmprobe.kill()

        # Distinguish metric types.
        metrics_lst = []
        for metr in range(0, len(self.metrics)):
            tmp = [item[metr] for item in collected]
            metrics_lst.append((self.metrics[metr], tmp))

        # Store to database.
        for item in metrics_lst:
            self.mongodb_con.insert_metrics(self.operator, self.machine, item[0], self.input_size, \
                    self.parameters, len(item[1]), item[1])

    def kill(self):
        """ Terminate the thread. """
        self.running = False


def main():
    """ Main function.
    Args:
        Taken from config.ini file.
    """

    # Parse configuration file.
    cfgparser = ConfigParser.ConfigParser()
    cfgparser.read("config.ini")
    host = cfgparser.get("Collector", "Monitored_Host")
    port = int(cfgparser.get("Collector", "Monitored_Port"))
    metrics = cfgparser.get("Gmond", "Metrics").split(",")

    # Connect to monitored machine.
    print "Attempting to connect to " + host + " at port " + str(port) + "."
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # Read command parameter file.
    with open(sys.argv[1]) as inputfile:
        params = inputfile.read().splitlines()
        print params
        for item in params:
            print item
            parts = item.split(" ")
            print parts
            # Initialization.
            machine = parts[0]
            operator = parts[1]
            input_size = parts[2]
            parameters = parts[3]
            # Make terminal command.
            command = ""
            for i in range(4, len(parts)):
                command = command + parts[i] + " "
            command = command[:-1]
            print "Command: " + command

            # Create collector.
            collector = Collector(metrics, operator, input_size, parameters, machine)

            print "Sending command...",
            sent = sock.send(command)
            if sent == 0:
                print "Socket connection broken."
            else:
                while 1:
                    msg = sock.recv(2048)

                    # Operator environment sends an "EXE_RDY" message when the operator
                    # is ready to execute.
                    if msg == "EXE_RDY":
                        print "Done!"
                        sent = sock.send("EXE_ACK")
                        if sent == 0:
                            print "Socket connection broken."
                            break
                        # Start collecting data.
                        print "Starting data collection."
                        collector.start()

                    # When operator finishes execution the environment sends an "EXE_END"
                    # message.
                    elif msg == "EXE_END":
                        collector.kill()
                        time.sleep(2)
                        print "Operator finished."
                        break

    # After all operators have been executed, terminate the connection.
    sent = sock.send("TERMINATE")
    if sent == 0:
        print "Socket connection broken."
    else:
        print "Terminating connection."
    sock.close()

if __name__ == "__main__":
    main()
