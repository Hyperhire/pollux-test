# ljw
import matplotlib.pyplot as plt
from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, quaternion2rotmat, rotmat2quaternion
from utils.graphics_utils import fov2focal
from utils.image_utils import resize_image
import os
import torch.nn.functional as F


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

#=================== apply padding in mask image (ljw, lsj) ========================

def load_dilated_mask_image(mask_path, iteration, resolution):

    padding_goal = 8 # defualt=12 the whole image size is 720*1200

    padding_size = int(padding_goal/resolution)

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

        # padding
        kernel = torch.ones((1, 3, 2*padding_size+1, 2*padding_size+1), dtype=torch.float32)  # 5x5 커널, 중앙 픽셀 포함
        resized_mask = resized_mask.unsqueeze(0).to(torch.float32)  # 배치 차원을 추가하고 float으로 변환

        # 2픽셀 확장 적용
        resized_mask = F.conv2d(resized_mask, kernel, padding=padding_size)  # padding을 2로 설정하여 테두리 처리
        resized_mask = resized_mask.squeeze(0) > 0  # 다시 원래 형태로 변환

        return resized_mask
#===========================================================================




        #=================== mask_gt and gt_image visualization (ljw, lsj)  ===============

        #print(f"viewpoint_cam.image_name : {viewpoint_cam.image_name}")
        #print(f"mask_gt : {mask_gt}")
        #print(f"mask_gt shape : {mask_gt.shape}")
        #print(f"gt_image : {gt_image}")
        #print(f"gt_image shape : {gt_image.shape}")
        ## Transfer tensors to CPU
        #mask_gt_np = mask_gt.cpu().numpy()
        #mask_gt_np = mask_gt_np.squeeze()
        #gt_image_np = gt_image.cpu().numpy()
        #gt_image_np = np.transpose(gt_image_np, (1, 2, 0))
        #plt.imshow(mask_gt_np, cmap='gray')
        #plt.savefig(f"./test/masked_gt_{viewpoint_cam.image_name}.png")
        #plt.close()    
        #plt.imshow(gt_image_np)
        #plt.savefig(f"./test/gt_image{viewpoint_cam.image_name}.png")
        #plt.close()    
        ##==================================================================================


