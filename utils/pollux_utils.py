# ljw
import matplotlib.pyplot as plt
from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, quaternion2rotmat, rotmat2quaternion
from utils.graphics_utils import fov2focal
from utils.image_utils import resize_image
import os

# ljw
def load_mask_image(mask_path, iteration, resolution):
    directory_path = mask_path
    file_format = "binary_mask_img_{}.jpg.npy"

    iteration = iteration.split('_')[-1]

    file_path = os.path.join(directory_path, file_format.format(iteration))
    if os.path.exists(file_path):
        binary_mask = np.load(file_path)
        binary_mask = binary_mask.astype(np.uint8)

        orig_h, orig_w, _ = binary_mask.shape

        new_resolution = (round(orig_h / resolution), round(orig_w / resolution))

        # Resize the image
        resized_mask = resize_image(torch.from_numpy(binary_mask).permute(2, 0, 1), new_resolution)
        resized_mask = (resized_mask > 0)

        return resized_mask