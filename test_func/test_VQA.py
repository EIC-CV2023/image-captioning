import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./BLIP_Mod'))
from PIL import Image
from models.blip_vqa import blip_vqa
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size,device):
    img_path = '/Users/vikimark/Documents/PythonFlow/EIC_Robocup/robocup2022-cv-image-captioning/dataset/w1.jpg' 
    raw_image = Image.open(img_path).convert('RGB')   

    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

image_size = 480
image = load_demo_image(image_size=image_size, device=device)     

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    
model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

question1 = 'What is color of the hair?'
question2 = 'What is color of the pants?'

with torch.no_grad():
    answer1 = model(image, question1, train=False, inference='generate')
    answer2 = model(image, question2, train=False, inference='generate')
    print(question1 + ' : ' + answer1[0])
    print(question2 + ' : ' + answer2[0])
