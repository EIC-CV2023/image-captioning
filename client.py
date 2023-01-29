import socket
import cv2
from custom_socket import CustomSocket

# time_interval = float(10000) # take input every 5 seconds

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0

host = socket.gethostname()
port = 10012
c = CustomSocket(host, port)
c.clientConnect()

while cap.isOpened():
    suc, frame = cap.read()
    if not suc:
        print("Can't read Frame")
        continue

    # Sent to sever
    img = cv2.resize(frame, (1280, 720))
    msg = c.req(img)
    print(msg)

    # Show client Frame
    cv2.imshow("client_cam", frame)
    if cv2.waitKey(1) == ord("q"):
        cap.release()
