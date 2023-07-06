import socket
import cv2
from custom_socket import CustomSocket
import json
import time

# time_interval = float(10000) # take input every 5 seconds

def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()
    
    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))


host = socket.gethostname()
port = 12303

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(list_available_cam(10))
cap.set(4, 720)
cap.set(3, 1280)

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Sent to sever
    

    # Show client Frame
    cv2.imshow("client_cam", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        cap.release()
    if key == ord("c"):
        cmd = int(input("1 for caption, 2 for ask: "))
        if cmd == 1:
            msg = c.req_with_command(frame, command={"task":"CAPTION"})
        if cmd == 2:
            questions = input("Input questions : ".split(","))
            msg = c.req_with_command(frame, command={"task":"ASK", "questions":questions})

        print(msg)
        cv2.waitKey()


cv2.destroyAllWindows()
