import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback

def main():
    HOST = socket.gethostname()
    PORT = 10011

    server = CustomSocket(HOST, PORT)
    server.startServer()

    while True:
        # Wait for connection from client
        conn, addr = server.sock.accept()
        print("Clent connected from", addr)

if __name__ == "__main__":
    main()