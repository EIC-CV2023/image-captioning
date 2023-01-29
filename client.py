import socket
import cv2
from custom_socket import CustomSocket

host = socket.gethostname()
port = 10011
c = CustomSocket(host, port)
c.clientConnect()