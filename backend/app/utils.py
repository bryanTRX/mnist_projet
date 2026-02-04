from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple

def canvas_to_grayscale_array(canvas_data: np.ndarray) -> np.ndarray:
    if np.issubdtype(canvas_data.dtype, np.floating):
        arr = (canvas_data * 255).astype(np.uint8)
    else:
        arr = canvas_data.astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        gray = np.mean(rgb, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    else:
        gray = arr

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def invert_if_needed(gray_arr: np.ndarray) -> np.ndarray:
    return gray_arr

def to_pil_and_resize(gray_arr: np.ndarray, size: Tuple[int,int]=(28,28)) -> Image.Image:
    img = Image.fromarray(gray_arr).convert("L")
    img = img.resize(size, resample=Image.BILINEAR)
    return img

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)

def preprocess_canvas_image(canvas_data: np.ndarray) -> torch.Tensor:
    gray = canvas_to_grayscale_array(canvas_data)
    gray = invert_if_needed(gray)         
    img = to_pil_and_resize(gray, (28,28))
    tensor = pil_to_tensor(img)
    return tensor

def get_preprocessed_pil(canvas_data: np.ndarray) -> Image.Image:
    gray = canvas_to_grayscale_array(canvas_data)
    gray = invert_if_needed(gray)
    img = to_pil_and_resize(gray, (28,28))
    return img
