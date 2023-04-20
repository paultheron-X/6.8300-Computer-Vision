# load the video data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse

#I downloaded the REDS data for video superresolution I want to crop a 512 512 video in the middle of each frame n python, how can I do it


def process_video(video_folder):
    # load the video (all the frames contained in the folder) read only one frame every 4
    video = []
    for i, img_name in enumerate(sorted(os.listdir(video_folder))):
        if i % 4 == 0:
            img = cv2.imread(os.path.join(video_folder, img_name))
            video.append(img)

    # image dimensions
    height, width, layers = video[0].shape

    # crop
    crop_size = 512

    # Define the crop coordinates
    crop_x = int((width - crop_size) / 2)
    crop_y = int((height - crop_size) / 2)

    # Crop the image
    cropped_video = [img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size] for img in video]


    # save the cropped video as a succession of images
    if 'GOOGLE_COLAB' in os.environ:
        saving_path = os.path.join(video_folder.replace('raw/train/train_orig', 'processed'))
    else:
        saving_path = os.path.join(video_folder.replace('raw', 'processed'))
    os.makedirs(saving_path, exist_ok=True)
    for i, img in enumerate(cropped_video):
        # save it like this: frame_0000.png, frame_0001.png, ..
        cv2.imwrite(os.path.join(saving_path, 'frame_{:04d}.png'.format(i)), img)


if __name__ == '__main__':
    # if the variable GOOGLE_COLAB is set to True, we are in Google Colab
    # and we need to mount the Google Drive
    if 'GOOGLE_COLAB' in os.environ:
        print('Running on Colab')
        raw_data_folder = '/content/drive/MyDrive/data_vision/raw/train/train_orig'
    else:
        print('Running locally')
        raw_data_folder = '../../../data/raw/train_orig'

    # iterate over all the videos with fancy progress bar
    for video_folder in tqdm(sorted(os.listdir(raw_data_folder))):
        process_video(os.path.join(raw_data_folder, video_folder))

