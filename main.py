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
from src.image_captioning import ask_model, paraphrase

questions_list = [
    "What is color of the hair?",
    "What is color of the shirt?"
]

print("Initializing age and gender model")
"""
TBC
"""
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
                age, gender = 22, 'male'

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
