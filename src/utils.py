from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def extract_grayscale_channel(canvas_data: np.ndarray) -> np.ndarray:
    return (canvas_data[:, :, 0] * 255).astype(np.uint8)

def to_pil_image(img_array: np.ndarray) -> Image.Image:
    return Image.fromarray(img_array).convert("L")

def resize_image(img: Image.Image, size: tuple[int,int]=(28,28)) -> Image.Image:
    return img.resize(size)

def to_tensor(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)

def preprocess_canvas_image(canvas_data: np.ndarray) -> torch.Tensor: 
    img_array   = extract_grayscale_channel(canvas_data)
    img         = to_pil_image(img_array)
    img_resized = resize_image(img)
    tensor      = to_tensor(img_resized)
    return tensor
