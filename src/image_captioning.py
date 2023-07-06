import os
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from src.load_VQA import transform_image
import numpy as np

def ask_model(model, image, questions_list):
    """Inference VQA model and return list of answers

    Args:
        model (BLIP_VQA): model to inference
        image (PIL.Image.Image): An image to be transformed
        questions_list (list of string): list of questions

    Returns:
        list of string: list of answers
    """
    transformed_image = transform_image(image)
    ans = []
    for question in questions_list:
        ans.append(*model(transformed_image, question, train=False, inference='generate'))
    
    return ans

def ask_model_with_prompt(model, image, questions_dict):
    """Inference VQA model and return list of answers

    Args:
        model (BLIP_VQA): model to inference
        image (PIL.Image.Image): An image to be transformed
        questions_list (list of string): list of questions

    Returns:
        list of string: list of answers
    """
    transformed_image = transform_image(image)
    ans = dict()
    for prompt in questions_dict:
        ans[prompt] = model(transformed_image, questions_dict[prompt], train=False, inference='generate')[0]
    
    return ans

def ask_model_with_questions(model, image, questions_list):
    """Inference VQA model and return list of answers

    Args:
        model (BLIP_VQA): model to inference
        image (PIL.Image.Image): An image to be transformed
        questions_list (list of string): list of questions

    Returns:
        list of string: list of answers
    """
    transformed_image = transform_image(image)
    ans = dict()
    for question in questions_list:
        ans[question] = model(transformed_image, question, train=False, inference='generate')[0]
    
    return ans

def paraphrase(answers):
    """Paraphrase answers to make them look more natural

    Args:
        gender (string): predicted gender of user
        age (int): predicted age of user
        qustions_list (list of string): question to ask VQA model
        captions (list of string): characteristic of user
    
    Returns:
        string: paraphrased text
    """
    text = ""
    gen = ""
    pro = ""
    poss = ""

    age = answers["age"]
    gender = answers["gender"]
    race = answers["race"]

    if gender:
        gen, pro, poss = ("man", "He", "His") if gender == "male" else ("woman", "She", "Her")
        text += f"{pro} is a {gen}. {pro} is {race}. {poss} apparent age is {age} years old. "


    for k, v in answers.items():
        # capt = 'a person posing for a picture'
        if k == "glasses":
            if v == "yes":
                text += f"{pro} is wearing glasses. "
            else:
                text += f"{pro} is not wearing glasses. "
        elif k not in ("age", "gender", "race"):
            text += f"{poss} {k} is {v}. "
            
    return text

def get_age_gender(frame, age_model, gender_model, haar_detector):
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
        # ---------------------------
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Female' if np.argmax(gender_class) == 0 else 'Male'

        return apparent_predictions, gender
    else:
        print("Can't detec your face.")
        return 0, ""