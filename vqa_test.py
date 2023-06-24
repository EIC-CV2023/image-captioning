import os
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from PIL import Image
from src.load_VQA import load_VQA_model
from src.image_captioning import ask_model, paraphrase


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()
    return list_cam


print("Initializing caption model")
model = load_VQA_model()
print("Done")

print(list_available_cam(10))

cap = cv2.VideoCapture(int(input("Cam index: ")))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Can't read frame")
        continue

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        cap.release()
    elif key == ord('s'):
        while True:
            question = input("Please type question: ")
            if question == 'q':
                break
            captions = ask_model(model, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), [question])
            print(captions)
            
cv2.destroyAllWindows()
