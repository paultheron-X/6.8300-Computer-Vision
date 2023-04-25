# load the video data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import argparse

#I downloaded the REDS data for video superresolution I want to crop a 512 512 video in the middle of each frame n python, how can I do it


def process_video(video_folder, crop = True, big_crop = True):
    # load the video (all the frames contained in the folder) read only one frame every 4
    video = []
    for i, img_name in enumerate(sorted(os.listdir(video_folder))):
        if i % 4 == 0:
            img_tensor = read_image(os.path.join(video_folder, img_name))
            video.append(img_tensor)

    # image dimensions
    channels, height, width = video[0].shape


    # Crop the image
    if crop:
        # Full img: 720x1280: crop size 512x512
        crop_size = 512
        # Define the crop coordinates
        crop_x = int((width - crop_size) / 2)
        crop_y = int((height - crop_size) / 2)
        cropped_video = [img_tensor[:, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size] for img_tensor in video]
        ppath = 'processed'
    elif big_crop:
        # The full image size is 720x1280, but the center crop is 504*896
        crop_size = (504, 896)  
        
        # Define the crop coordinates
        crop_x = int((width - crop_size[1]) / 2)
        crop_y = int((height - crop_size[0]) / 2)
        
        cropped_video = [img_tensor[:, crop_y:crop_y+crop_size[0], crop_x:crop_x+crop_size[1]] for img_tensor in video]
        ppath = 'processed/big_crop'
    else:
        cropped_video = video
        ppath = 'processed/fimg'


    # save the cropped video as a succession of images
    if 'GOOGLE_COLAB' in os.environ:
        saving_path = os.path.join(video_folder.replace('raw/train/train_orig', ppath))
    else:
        saving_path = os.path.join(video_folder.replace('raw/train_orig', ppath))
    os.makedirs(saving_path, exist_ok=True)
    for i, img in enumerate(cropped_video):
        # save it like this: frame_0000.png, frame_0001.png, ..
        img_name = os.path.join(saving_path, f'frame_{i:04d}.png')
        save_image(img/255, img_name)

if __name__ == '__main__':
    # if the variable GOOGLE_COLAB is set to True, we are in Google Colab
    # and we need to mount the Google Drive
    if 'GOOGLE_COLAB' in os.environ:
        print('Running on Colab')
        raw_data_folder = '/content/drive/MyDrive/data_vision/raw/train/train_orig'
    else:
        print('Running locally')
        raw_data_folder = 'data/raw/train_orig'

    # iterate over all the videos with fancy progress bar
    print('Processing videos with big crop')
    for video_folder in tqdm(sorted(os.listdir(raw_data_folder))):
        process_video(os.path.join(raw_data_folder, video_folder), crop=False, big_crop=True)
    
    print('Processing videos with crop')
    for video_folder in tqdm(sorted(os.listdir(raw_data_folder))):
        process_video(os.path.join(raw_data_folder, video_folder), crop=True, big_crop=False)
    
    print('Processing videos with full image')
    for video_folder in tqdm(sorted(os.listdir(raw_data_folder))):
        process_video(os.path.join(raw_data_folder, video_folder), crop=False, big_crop=False)

