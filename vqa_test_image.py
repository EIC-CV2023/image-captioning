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


print("Initializing caption model")
model = load_VQA_model()
print("Done")

PATH = "test_img"

folder = os.listdir(PATH)
im_id = 0

while True:
    img = cv2.imread(f"{PATH}/{folder[im_id]}")

    cv2.imshow("frame", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
                  break
    if key == ord('a'):
        question = input("Please type question: ")
        if question == 'q':
            break
        captions = ask_model(model, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [question])
        print(captions)
    if key == ord('n'):
           im_id += 1
           
            
cv2.destroyAllWindows()
