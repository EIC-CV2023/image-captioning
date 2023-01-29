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
from human_crop import crop
from load_VQA import load_VQA_model, transform_image
from image_captioning import ask_model, paraphrase

questions_list = [
    'What is color of the hair?',
    'What is color of the pants?']
# path file to test. Change these to inference
path = '/Users/vikimark/Documents/PythonFlow/EIC_Robocup/26504621-two-couples-of-young-casual-fashion-people-posing-for-the-camera-women-looking-at-their-men.jpg'

# dummy age_gender model
print("Initializing age and gender model...")
"""
TBC
"""
print("Done")

print("Initializing caption model")
model = load_VQA_model() # load model in main due to reproductivity

frame = cv2.imread(path)
crop_image = crop(frame)
if crop_image:
    whole_image = crop_image['whole']
    img = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
    # ------- debug ------------------
    # cv2.imshow('image', whole_image)
    # cv2.waitKey(0)

    # dummy
    age, gender = 22, 'male'
    captions = ask_model(model, Image.fromarray(img), questions_list)
    print(paraphrase(gender, age, questions_list, captions))
else:
    print("No human detected.")