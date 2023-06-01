import os
import sys
import numpy as np
import cv2

def get_age_gender(frame):
    print(f"\nAnalysing age and gender....\n")

    output_indexes = np.array([i for i in range(0, 101)])

    if frame.size != 0:
        # frame = crop["head"]
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = np.array(haar_detector.detectMultiScale(gray, 1.3, 5))
        if faces.size != 0:
            for face in faces:
                x, y, w, h = face
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]
                break
            else:
                print('can not detect face in for')
                return 0, ""

        else:
            print('can not detect face in if')
            return 0, ""

        # age model is a regular vgg and it expects (224, 224, 3) shape input

        detected_face = cv2.resize(detected_face, (224, 224))
        img_blob = cv2.dnn.blobFromImage(detected_face)  # caffe model expects (1, 3, 224, 224) shape input
        # ---------------------------
        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        apparent_predictions = round(np.sum(age_dist * output_indexes))
        print("Apparent age: ", apparent_predictions)
        # ---------------------------
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Female' if np.argmax(gender_class) == 0 else 'Male'
        print("Gender: ", gender)

        return apparent_predictions, gender
    else:
        print("Can't detec your face.")
        return 0, ""


print("Initializing age and gender model...")
haar_detector = cv2.CascadeClassifier("age_gender_recog/haarcascade_frontalface_default.xml")
age_model = cv2.dnn.readNetFromCaffe("age_gender_recog/age.prototxt",
                                     "age_gender_recog/age_caffe.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("age_gender_recog/gender.prototxt", "age_gender_recog/gender_caffe.caffemodel")
print("Done.")

path = 'test_img/m1.jpg'
frame = cv2.imread(path)

age, gender = get_age_gender(frame)
print(age, gender)

