import sys
sys.path.append("./BLIP_Mod")
from PIL import Image
from models.blip_vqa import blip_vqa
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_VQA_model(image_size=480):
    """Load VQA model and return it

    Args:
        image_size (int, optional): Size of the images (Square images)

    Returns:
        models.blip_vqa.BLIP_VQA: VQA's model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    return model

def transform_image(raw_image, image_size=480):
    """Transform and normalize image to the way VQA was trained

    Args:
        raw_image (PIL.Image.Image): An Raw image loaded from Image function
        image_size (int, optional): Size of the images (Square images)

    Returns:
        torch.Tensor: Transformed and normalized image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w, h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image
