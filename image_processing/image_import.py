import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

digits_name = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

def import_image(filepath: Path, width: int = 28, height: int = 28) -> np.ndarray:
    image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized / 255.0


def import_images(instances: int, start: int, filepath: Path, width: int = 28, height: int = 28) -> tuple[np.ndarray, np.ndarray]:
    train_x_list = []
    train_y_list = []
    for i in range(10):
        folder = filepath / str(i)
        for k in range(start, start + instances):
            image_path = folder / f"{digits_name[i]}_full ({k}).jpg"
            img = import_image(image_path, width, height)

            if img is not None:
                train_x_list.append(img.flatten())
                ret = []
                for l in range(10):
                    if l == i:
                        ret.append(1)
                    else:
                        ret.append(0)
                train_y_list.append(ret)

    train_x = np.array(train_x_list)
    train_y = np.array(train_y_list)
    return (train_x, train_y)