import os
import sys
# sys.path.append(os.path.abspath('.'))
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from PIL import Image
from src.load_VQA import load_VQA_model
from src.image_captioning import ask_model, paraphrase, get_age_gender, ask_model_with_prompt

# questions_list = [
#     "What is color of the hair?",
#     "What is color of the shirt?"
# ]
questions_dict = {
    "age" : "How old is the person?",
    "gender" : "What gender is the person?",
    "race" : "What is the person race?",
    "hair color" : "What is color of the hair?",
    "shirt color" : "What is color of the shirt?",
    "glasses" : "Does the person wear glasses?"
}



def main():
    # print("Initializing age and gender model")
    # haar_detector = cv2.CascadeClassifier("age_gender_recog/haarcascade_frontalface_default.xml")
    # age_model = cv2.dnn.readNetFromCaffe("age_gender_recog/age.prototxt",
    #                                     "age_gender_recog/age_caffe.caffemodel")
    # gender_model = cv2.dnn.readNetFromCaffe("age_gender_recog/gender.prototxt", "age_gender_recog/gender_caffe.caffemodel")
    # print("Done")

    print("Initializing caption model")
    model = load_VQA_model()
    print("Done")

    HOST = socket.gethostname()
    PORT = 12303

    server = CustomSocket(HOST, PORT)
    server.startServer()

    while True:
        # Wait for connection from client
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        while True:
            try:
                data = server.recvMsg(
                    conn, has_splitter=True)
                frame_height, frame_width, frame = data
                
                # crop_image = crop(frame) MediaPipe is no longer used, and input was image that has already been cropped.

                res = dict()
                if frame.size != 0:
                    # whole_image = crop_image['whole']
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # dummy
                    """
                    This space is for improving age & gender to be more precise by giving image to be specific to head section
                    # frame = head_crop_image...
                    """
                    # age, gender = get_age_gender(frame, age_model, gender_model, haar_detector)
                    # ans["age"] = age
                    # ans["gender"] = gender

                    answers = ask_model_with_prompt(model, Image.fromarray(frame), questions_dict)
                    
                    res = answers
                    
                    res["answer"] = paraphrase(answers)
                else:
                    res["answer"] = ""

                server.sendMsg(conn, json.dumps(res))
                
                #TODO send an image to WalkieUI
                # # Display the image
                # cv2.imshow('Image', frame)
                # cv2.waitKey(6000)  # waits for 6000 ms (6 seconds)

                # # It's a good practice to destroy windows after you are done
                # cv2.destroyAllWindows()

            except Exception as e:
                traceback.print_exc()
                # print(e)
                print("Connection Closed")
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
