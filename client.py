import socket
import cv2
from custom_socket import CustomSocket

host = socket.gethostname()
port = 10012
c = CustomSocket(host, port)
c.clientConnect()

data = cv2.imread("test_img/test1.jpg")
data = cv2.resize(data, (1280, 720))

msg = c.req(data)
print(msg)
