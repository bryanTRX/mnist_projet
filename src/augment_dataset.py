import os
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import random
from tqdm import tqdm


def augment_image(img: Image.Image) -> Image.Image:
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=0)

    max_shift = 2
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    img = ImageChops.offset(img, shift_x, shift_y)

    scale = random.uniform(0.9, 1.1)
    w, h = img.size
    img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    img_final = Image.new("L", (28, 28), 0)
    paste_x = max((28 - img.size[0]) // 2, 0)
    paste_y = max((28 - img.size[1]) // 2, 0)
    img_final.paste(img, (paste_x, paste_y))

    if random.random() < 0.3:
        img_final = img_final.filter(ImageFilter.GaussianBlur(radius=1))

    if random.random() < 0.3:
        arr = np.array(img_final)
        arr = np.clip(arr + np.roll(arr, 1, axis=0) + np.roll(arr, 1, axis=1), 0, 255)
        img_final = Image.fromarray(arr.astype(np.uint8))

    return img_final

def augment_dataset(input_dir: str, output_dir: str, n_augment: int = 5):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(input_class_dir), desc=f"Processing class {class_name}"):
            img_path = os.path.join(input_class_dir, img_name)
            img = Image.open(img_path).convert("L")

            base_name, ext = os.path.splitext(img_name)
            img.save(os.path.join(output_class_dir, f"{base_name}_orig{ext}"))
            for i in range(n_augment):
                aug_img = augment_image(img)
                aug_img.save(os.path.join(output_class_dir, f"{base_name}_aug{i}{ext}"))

    print(f"Dataset augmenté créé dans {output_dir}")

if __name__ == "__main__":
    augment_dataset(input_dir="data/train", output_dir="data/train_test", n_augment=3)