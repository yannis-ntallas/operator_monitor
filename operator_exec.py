#!/usr/bin/env python

""" Operator Execution Environment.

The following python module runs on the slave node
of the system and synchronizes the operator execution with the monitoring or collection
process. """

import socket
import subprocess

__author__ = "Ioannis Ntallas"
__copyright__ = "Copyright 2018, Ioannis Ntallas. All rights reserved."
__license__ = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"
__version__ = "0.1"
__maintainer__ = "Ioannis Ntallas"
__email__ = "ynts@outlook.com"
__status__ = "Development"

MAX_CONN = 5

def run_operator(command):
    """  Spawns a subprocess for the operator.

    Args:
        command (str) : operator command and arguments.

    """

    parts = command.split(" ")
    # Spawn subprocess.
    operator = subprocess.Popen(parts, \
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for operator to finish.
    exit_code = operator.wait()
    if exit_code != 0:
        print "Operator failure."
    return exit_code


def main():
    """ Main function. """

    # Create a server socket.
    print "Creating server socket."
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        serversocket.bind(('', 8888))
    except socket.error as msg:
        print "SOCKET BINDING ERROR: " + msg
    serversocket.listen(MAX_CONN)

    # Accept incoming execution request.
    print "Waiting for connections..."
    (clientsocket, address) = serversocket.accept()
    print "Monitor connected!"

    while 1:
        msg = clientsocket.recv(2048)
        if msg == "TERMINATE":
            print "Terminating operator environment."
            serversocket.close()
            break
        else:
            # Notify that operator will begin execution.
            sent = clientsocket.send("EXE_RDY")
            if sent == 0:
                print "Ready to execute."
                print "Socket connection broken."
                break

            # Run the operator.
            responce = clientsocket.recv(2048)
            if responce == "EXE_ACK":
                print "Operator running..."
                exit_code = run_operator(msg)
                print exit_code

            # Notify that operator has finished.
            sent = clientsocket.send("EXE_END")
            print "Done!"
            if sent == 0:
                print "Socket connection broken."
                break

if __name__ == "__main__":
    main()
