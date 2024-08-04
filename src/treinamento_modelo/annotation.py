import numpy as np
from PIL import Image
import os

def load_images_from_folder(folder, target_size=None):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  
            images.append(img_array)
    return np.array(images)

rgb_folder = 'dados/blocos'
segmented_folder = 'dados/segmentadas'
save_folder = 'dados/modelo/'

x_train = load_images_from_folder(rgb_folder)
y_train = load_images_from_folder(segmented_folder)

y_train = (y_train > 0).astype(np.float32)

np.save(os.path.join(save_folder, 'x_train.npy'), x_train)
np.save(os.path.join(save_folder, 'y_train.npy'), y_train)
