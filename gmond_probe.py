#!/usr/bin/env python

""" Gmond probe program.

This program accesses ganglia's gmond via a tcp socket and returns an xml file containing all the
information gmond can provide.

"""
import threading
import socket
import time
import Queue
import ConfigParser
import xml.etree.ElementTree as ET

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

class GmondProbe(threading.Thread):
    """ Gmond Probe class.

    Open a socket to the gmond server and get data. To get a sample xml file run the command:
    $ telnet 127.0.0.1 8649 >> out.txt

    Attributes:
        host (str) : The IP address of gmond.
        port (int) : gmond's default port.
        running (bool) : Used to check whether the program is being terminated.
        delay (int) : Sets the delay between each request.
        queue (queue) : A shared queue from which another program can read the metrics.
        metrics (str list) : Specifies which metric types will be saved.

    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.cfgparser = ConfigParser.ConfigParser()
        self.cfgparser.read("config.ini")
        self.host = self.cfgparser.get("Gmond", "Host")
        self.port = int(self.cfgparser.get("Gmond", "Port"))
        self.running = True
        self.delay = int(self.cfgparser.get("Gmond", "Delay"))
        self.queue = Queue.Queue()
        self.metrics = self.cfgparser.get("Gmond", "Metrics").split(",")

    def run(self):
        while self.running is True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            xml_file = []
            while True:
                data = sock.recv(1024)
                if not data:
                    if xml_file:
                        self.parse_xml(xml_file)
                        del xml_file
                        time.sleep(self.delay)
                    break
                else:
                    xml_file.append(data)

    def parse_xml(self, xml_file):
        """ Get the xml from gmond and parse it to a tree. Then extract the attributes we need and
        save it in the shared queue as tuples of the following format:

        (host_name, (metr1, metr2, .. metrN))

        Args:
            xml_file : the xml file provided by gmond.

        """
        root = ET.fromstringlist(xml_file)
        for cluster in root:
            for host in cluster:
                metr = [None] * len(self.metrics)
                for metric in host:
                    name = metric.get('NAME')
                    if name in self.metrics:
                        value = metric.get('VAL')
                        # xml elements not in the same order fix
                        index = self.metrics.index(name)
                        metr[index] = value
                # Return values for each machine as a tuple with a queue.
                self.queue.put((host.get('NAME'), metr))

    def kill(self):
        """ Terminate the thread. """
        self.running = False

def main():
    """ Main function for testing purposes. """
    gmprobe = GmondProbe()
    gmprobe.start()
    test = gmprobe.queue.get()
    print test[0]
    print test[1]
    test = gmprobe.queue.get()
    print test[0]
    print test[1]
    gmprobe.kill()

if __name__ == "__main__":
    main()
