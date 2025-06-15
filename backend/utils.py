import numpy as np
from PIL import Image

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    return img_array
