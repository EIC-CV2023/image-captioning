import os
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from src.custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from PIL import Image
from src.load_VQA import load_VQA_model
from src.image_captioning import ask_model, paraphrase, get_age_gender

questions_list = [
    "What is color of the hair?",
    "What is color of the shirt?"
]

print("Initializing age and gender model")
haar_detector = cv2.CascadeClassifier("age_gender_recog/haarcascade_frontalface_default.xml")
age_model = cv2.dnn.readNetFromCaffe("age_gender_recog/age.prototxt",
                                     "age_gender_recog/age_caffe.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("age_gender_recog/gender.prototxt", "age_gender_recog/gender_caffe.caffemodel")
print("Done")

print("Initializing caption model")
model = load_VQA_model()
print("Done")

HOST = socket.gethostname()
PORT = 10012

server = CustomSocket(HOST, PORT)
server.startServer()

while True:
    # Wait for connection from client
    conn, addr = server.sock.accept()
    print("Client connected from", addr)

    while True:
        try:
            data = server.recvMsg(conn)
            img = np.frombuffer(data, dtype=np.uint8).reshape(720, 1280, 3)
            # crop_image = crop(img) MediaPipe is no longer used, and input was image that has already been cropped.
            if img.size != 0:
                # whole_image = crop_image['whole']
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # dummy
                """
                This space is for improving age & gender to be more precise by giving image to be specific to head section
                # img = head_crop_image...
                """
                age, gender = get_age_gender(img, age_model, gender_model, haar_detector)

                captions = ask_model(model, Image.fromarray(img), questions_list)
                ans = {"answer":paraphrase(gender, age, questions_list, captions)}
            else:
                ans = {"anser":""}
            server.sendMsg(conn, json.dumps(ans))

            # Show image
            # cv2.imshow("Server Cam", img)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        except Exception as e:
            traceback.print_exc()
            # print(e)
            print("Connection Closed")
            break
    cv2.destroyAllWindows()
