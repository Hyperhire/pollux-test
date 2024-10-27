import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import torch.nn.functional as F
from PIL import Image
# from segment_anything import sam_model_registry, SamPredictor
import argparse
from tqdm import tqdm

main_img_root = '/root/workspace/src/preprocessing'

# Framing video to images
def frame_video(video_name):
    # Create a VideoCapture object
    video_path = f"{main_img_root}/input_video/" + video_name
    cap = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video in {video_path}")
        return
    print(f"Video in {video_path} was opened")

    # Directory where the images will be saved
    save_dir = f"{main_img_root}/frames_from_video/" + video_name.split(".")[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print("Already frame_video Done!")
        return
    # Frame counter
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            print("Can't receive frame.")
            break  # Exit loop when no more frames to read or if there's an error

        # Save frame as JPG file
        frame_filename = os.path.join(save_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        frame_count += 1

    # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    print("Video processing complete. All frames are extracted and saved.")
    return 

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# Background segmentation function for one image (input: image_path), this function segemnts the feature that contains the center point in the image.
from lang_sam import LangSAM
model = LangSAM()
def segmentation_usingLangSAM(image_path):

    image_pil = Image.open(image_path).convert("RGB")
    text_prompt = "cloth."
    results = model.predict([image_pil], [text_prompt])

    mask = results[0]['masks'][0]

    result_mask = (mask > 0.5).astype(np.uint8)

    image_np = np.array(image_pil)

    result_image_np = image_np * result_mask[:, :, np.newaxis]

    result_mask = np.repeat(result_mask[:, :, np.newaxis], 3, axis=2)

    result_image = Image.fromarray(result_image_np)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 1, axis=2)

    return result_image, None, result_mask

def segmentation_usingSAM(image_path, predictor):
    img = Image.open(image_path)
    predictor.set_image(np.array(img))

    width = img.size[0]
    hight = img.size[1]

    # Select points those should be and shouldn't be in the segmentation 
    input_point = np.array([[width/2, hight/2], [width/12, hight/12], [width*11/12, hight*11/12],  [width*11/12, hight/12], [width/12, hight*11/12]])
    input_label = np.array([1, 0, 0, 0, 0])
    
    masks, quality_scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    max_mask_coverage = 0
    selected_mask = None
    for mask in masks:
        coverage_area = mask.sum()
        if coverage_area > max_mask_coverage:
            max_mask_coverage = coverage_area
            selected_mask = mask
    
    if masks.shape[0] > 0:  # Check if any masks were predicted
        mask = selected_mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand mask to 3 channels
        foreground = np.array(img) * mask_3d
    else:
        foreground = np.zeros_like(np.array(img))  # No object detected, return empty mask
    
    result_img = Image.fromarray(foreground.astype(np.uint8))
    result_mask = Image.fromarray((mask_3d*255).astype(np.uint8))
    return result_img, None, result_mask
    
# Segment all images in the frames that is in the video
def segment_frames_from_video(video_name, frame_num = 150):
    # sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    # sam.to(device = "cuda")
    # predictor = SamPredictor(sam)
    input_folder = f"{main_img_root}/frames_from_video/" + video_name.split(".")[0]
    save_dir = f"{main_img_root}/segmented_frames/" + video_name.split(".")[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print("Already SAM Done!")
        return
    binary_mask_dir = save_dir + "/binary_mask"
    if not os.path.exists(binary_mask_dir):
        os.makedirs(binary_mask_dir)
    files =os.listdir(input_folder)
    sorted_files = sorted(files, key=str.lower)

    per_frame = int(len(sorted_files)/int(frame_num))
    count = 0
    mask_size_threshold = 500
    # for filename in sorted(sorted_files):
    for filename in tqdm(sorted_files, desc="Segmenting frames"):
        if count % per_frame == 0 & filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            # result_img, mask_img, mask = segmentation_usingSAM(input_path, predictor)
            result_img, mask_img, mask = segmentation_usingLangSAM(input_path)
            mask_size = np.sum(mask)
            if mask_size >= mask_size_threshold:
                result_img.save(os.path.join(save_dir, filename))
                np.save(os.path.join(binary_mask_dir, 'binary_mask_' + filename.split('.')[0] + '.npy'), mask)
        count+=1
    print("Frame segmentation is completed. All frames are segmented and image, masks are saved.")

# Make background white using the SAM numpy mask
def white_background_frames(video_name):
    input_folder = f"{main_img_root}/frames_from_video/" + video_name.split(".")[0]
    mask_folder = f"{main_img_root}/segmented_frames/" + video_name.split(".")[0] + "/binary_mask"
    save_dir = f"{main_img_root}/white_background_frames/" + video_name.split(".")[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    files = os.listdir(input_folder)
    sorted_files = sorted(files, key=str.lower)
    
    for filename in sorted(sorted_files):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            mask_path = os.path.join(mask_folder, 'binary_mask_' + filename.split('.')[0] + '.npy')
            
            if os.path.exists(mask_path):
                img = Image.open(input_path).convert("RGB")
                mask_3d = np.load(mask_path)
                # mask_3d = np.repeat(mask_3d[:, :, np.newaxis], 3, axis=2)
                
                # Create white background
                white_background = np.ones_like(np.array(img)) * 255
                
                # Apply the mask to the image and combine with white background
                foreground = np.array(img) * mask_3d
                result = white_background * (1 - mask_3d) + foreground
                
                result_img = Image.fromarray(result.astype(np.uint8))
                result_img.save(os.path.join(save_dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment objects in an image using SAM.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--frame_num", type=str, required=True, help="Number of image frames.")
    
    args = parser.parse_args()
    frame_video(args.img_path)
    segment_frames_from_video(args.img_path, args.frame_num)
    white_background_frames(args.img_path)
