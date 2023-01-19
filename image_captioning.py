import os
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from PIL import Image
from human_crop import crop
from load_VQA import load_VQA_model, transform_image

def ask_model(model, image, questions_list):
    """Inference VQA model and return list of answers

    Args:
        model (): model to inference
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
        gen, pro, poss = ("man", "He", "His") if gender == "Male" else ("woman", "She", "Her")
        text += f"{pro} is a {gen}. {poss} apparent age is {age} years old. "

    if questions_list:
        for i, question in enumerate(questions_list):
            # capt = 'a person posing for a picture'
            if 'color' in question:
                text += f"{poss} {question.split()[-1][:-1]} color is {captions[i]}. "
            else:
                text += f"I saw {poss} {question.split()[-1][:-1] is {captions[i]}}"
            
    return text

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