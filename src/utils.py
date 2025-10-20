from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple

def canvas_to_grayscale_array(canvas_data: np.ndarray) -> np.ndarray:
    """
    Prend canvas_result.image_data (H x W x C), retourne un array 2D uint8 (0-255)
    - supporte float [0..1] ou uint8 [0..255]
    - gère RGBA (4 canaux) ou RGB (3 canaux) ou déjà 1 canal
    """
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
    """
    Inverse les couleurs si le fond est clair et le trait sombre.
    On veut : trait BLANC (255) sur fond NOIR (0) — comme MNIST.
    On considère que si la moyenne > 127 alors fond clair -> on inverse.
    """
    mean = gray_arr.mean()
    if mean > 127:

        return 255 - gray_arr
    return gray_arr

def to_pil_and_resize(gray_arr: np.ndarray, size: Tuple[int,int]=(28,28)) -> Image.Image:
    """
    Convertit en PIL Image (mode 'L') puis resize en conservant la meilleure interpolation.
    """
    img = Image.fromarray(gray_arr).convert("L")
    img = img.resize(size, resample=Image.BILINEAR)
    return img

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convertit une PIL Image (L) en tensor shape (1,1,H,W) float32 [0,1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    return transform(img).unsqueeze(0)

def preprocess_canvas_image(canvas_data: np.ndarray) -> torch.Tensor:
    """
    Pipeline complet : canvas_data -> tensor (1,1,28,28) prêt pour le modèle.
    """
    gray = canvas_to_grayscale_array(canvas_data)
    gray = invert_if_needed(gray)         
    img = to_pil_and_resize(gray, (28,28))
    tensor = pil_to_tensor(img)           
    return tensor

def get_preprocessed_pil(canvas_data: np.ndarray) -> Image.Image:
    """
    Retourne la PIL 28x28 utilisée pour afficher/debug (après inversion/resize).
    Utile pour afficher côté Streamlit.
    """
    gray = canvas_to_grayscale_array(canvas_data)
    gray = invert_if_needed(gray)
    img = to_pil_and_resize(gray, (28,28))
    return img
