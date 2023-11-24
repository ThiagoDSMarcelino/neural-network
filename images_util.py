import os
from typing import List
from PIL import Image

def __is_image_file(file_path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def __is_image_corrupted(file_path: str):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError):
        return True

def find_images(directory: str) -> List[str]:
    images = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if __is_image_file(file_path):
                images.append(file_path)

    return images

def delete_corrupted_images(path: str):
    images = find_images(path)
    corrupted_images = list(filter(__is_image_corrupted, images))

    if corrupted_images:
        print("Corrupted images found:")
        for img_path in corrupted_images:
            os.remove(img_path)
            print(img_path)
    else:
        print("None corrupted image has found.")