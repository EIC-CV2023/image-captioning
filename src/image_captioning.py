import os
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from src.load_VQA import transform_image

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

def paraphrase(gender="", age=0, questions_list=[], captions=None, ):
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

    if gender:
        gen, pro, poss = ("man", "He", "His") if gender == "male" else ("woman", "She", "Her")
        text += f"{pro} is a {gen}. {poss} apparent age is {age} years old. "

    if questions_list:
        for i, question in enumerate(questions_list):
            # capt = 'a person posing for a picture'
            if 'color' in question:
                text += f"{poss} {question.split()[-1][:-1]} color is {captions[i]}. "
            else:
                text += f"I saw {poss} {question.split()[-1][:-1] is {captions[i]}}"
            
    return text