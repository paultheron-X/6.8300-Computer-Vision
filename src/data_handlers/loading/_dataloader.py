import os
import sys
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)  # for testing this file with main and so that utils can be imported

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from torchvision.io import read_image

import glob
import random
from tqdm import tqdm

from utils.logging_config import logger_setup
from utils.arguments_parser import data_loading_parser


class VideoDataset(Dataset):
    def __init__(
        self,
        lr_data_dir,
        hr_data_dir,
        is_test=False,
        is_small_test=False,
        rolling_window=5,
        **kwargs,
    ):
        if not is_test:  # always put the training folder as input
            self.lr_data_dir = lr_data_dir
            self.hr_data_dir = hr_data_dir
        else:
            self.lr_data_dir = lr_data_dir.replace("train", "val")
            self.hr_data_dir = hr_data_dir.replace("train", "val")

        if rolling_window % 2 == 0:
            raise ValueError("Rolling window must be odd")

        self.is_test = is_test

        self.rolling_window = rolling_window
        self.patch_size = kwargs.get("patch_size", 64)

        self.total_keys = sorted(os.listdir(self.lr_data_dir))
        self.total_keys = [
            i for i in self.total_keys if i not in ["downsampled", "fimg", "big_crop"]
        ]

        small_val_keys = ["000", "010", "020", "029"]
        if is_small_test:
            self.keys = small_val_keys
        else:
            self.keys = self.total_keys
        
        self.files_per_key = {
                key: sorted(
                    os.listdir(os.path.join(self.lr_data_dir, key))
                )  # get all the files in the key directory
                for key in self.keys
            }
        if not kwargs.get("skip_frames", 0):
            self.frames = {
                key: [self.files_per_key[key][i:i+self.rolling_window] for i in range(0, len(self.files_per_key[key])-self.rolling_window+1, 1)]
                for key in self.keys
            }
            self.num_files_per_key = {
                key: len(self.frames[key]) for key in self.keys
            }
        else:
            # we will skip some frames in the dataset:
            # example: if we have a rolling window of 5, we would do, with 10 images
            # (f0 f1 f2 f3 f4) , then (f3 f4 f5 f6 f7) then (f6 f7 f8 f9)
            self.frames = {
                key: [self.files_per_key[key][i:i+self.rolling_window] for i in range(0, len(self.files_per_key[key]), self.rolling_window-2)]
                for key in self.keys
            }
            # if the last frame length is not the same as the rolling window, we remove it
            for key in self.keys:
                if len(self.frames[key][-1]) != self.rolling_window:
                    self.frames[key] = self.frames[key][:-1]
            
            self.num_files_per_key = {
                key: len(self.frames[key])
                for key in self.keys
            }
        self.len = sum(self.num_files_per_key.values())
        
        #print("Dataset length: ", self.len)
        #print("Keys: ", self.keys)
        #print("Number of files per key: ", self.num_files_per_key)
        #print("Frames per key: ", self.frames)
            

    def __len__(self):
        # this dataset will extract batch of rolling_window frames from each video
        # will take the keys sequentially
        return self.len

    def __getitem__(self, index):
        # 1. get the key and the index of the file in the key
        # get the key and the index of the file in the key
        key, frame_idx = self.get_key_and_frame_idx(index)
        # get the file names of the rolling window
        file_names = self.frames[key][frame_idx]
        
        # 2. read the images part 
        # gt_image
        gt_images = [
            read_image(os.path.join(self.hr_data_dir, key, file_name)) / 255
            for file_name in file_names
        ]
        gt_images_tensor = torch.stack(gt_images)
        # read the images
        lr_images = [
            read_image(os.path.join(self.lr_data_dir, key, file_name)) / 255
            for file_name in file_names
        ]
        # stack the images
        lr_images_tensor = torch.stack(lr_images)  # (t, c, h, w)

        lr_images_tensor = lr_images_tensor[
            :, :, 4:, :
        ]  # TODO: Still necessary with pad size
        gt_images_tensor = gt_images_tensor[:, :, 16:, :]

        if not self.is_test:
            lr_images_tensor, gt_images_tensor = self.transform(
                gt_images_tensor, lr_images_tensor
            )
            pass

        return lr_images_tensor, gt_images_tensor

    def get_key_and_frame_idx(self, index):
        # get the key and the index of the file in the key
        # all the keys have the same number of frame per key    
        for key in self.keys:
            if index < self.num_files_per_key[key]:
                return key, index
            else:
                index -= self.num_files_per_key[key]
        raise ValueError("Index out of range")

    def transform(self, gt_seq, lq_seq):
        gt_transformed, lq_transformed = pair_random_crop_seq(
            gt_seq, lq_seq, patch_size=self.patch_size
        )
        gt_transformed, lq_transformed = pair_random_flip_seq(
            gt_transformed, lq_transformed, p=0.5
        )
        gt_transformed, lq_transformed = pair_random_transposeHW_seq(
            gt_transformed, lq_transformed, p=0.5
        )
        return gt_transformed, lq_transformed


def pair_random_crop_seq(hr_seq, lr_seq, patch_size, scale_factor=4):
    """crop image pair for data augment
    Args:
        hr (Tensor): hr images with shape (t, c, 4h, 4w).
        lr (Tensor): lr images with shape (t, c, h, w).
        patch_size (int): the size of cropped image
    Returns:
        Tensor, Tensor: cropped images(hr,lr) of shape (t, c, patch_size, patch_size) and (t, c, patch_size*scale, patch_size*scale)
    """
    seq_lenght = lr_seq.size(dim=0)
    gt_transformed = torch.empty(
        seq_lenght, 3, patch_size * scale_factor, patch_size * scale_factor
    )
    lq_transformed = torch.empty(seq_lenght, 3, patch_size, patch_size)
    i, j, h, w = T.RandomCrop.get_params(
        lr_seq[0], output_size=(patch_size, patch_size)
    )
    gt_transformed = T.functional.crop(
        hr_seq, i * scale_factor, j * scale_factor, h * scale_factor, w * scale_factor
    )
    lq_transformed = T.functional.crop(lr_seq, i, j, h, w)
    return gt_transformed, lq_transformed


def pair_random_flip_seq(sequence1, sequence2, p=0.5, horizontal=True, vertical=True):
    """flip image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being flipped.
            Default: 0.5
        horizontal (bool): Store `False` when don't flip horizontal
            Default: `True`.
        vertical (bool): Store `False` when don't flip vertical
            Default: `True`.
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length = sequence1.size(dim=0)
    # Random horizontal flipping
    hfliped1 = sequence1.clone()
    hfliped2 = sequence2.clone()
    if horizontal and random.random() > 0.5:
        hfliped1 = T.functional.hflip(sequence1)
        hfliped2 = T.functional.hflip(sequence2)

    # Random vertical flipping
    vfliped1 = hfliped1.clone()
    vfliped2 = hfliped2.clone()
    if vertical and random.random() > 0.5:
        vfliped1 = T.functional.vflip(hfliped1)
        vfliped2 = T.functional.vflip(hfliped2)
    return vfliped1, vfliped2


def pair_random_transposeHW_seq(sequence1, sequence2, p=0.5):
    """crop image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being cropped.
            Default: 0.5
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length = sequence1.size(dim=0)
    transformed1 = sequence1.clone()
    transformed2 = sequence2.clone()
    if random.random() > 0.5:
        transformed1 = torch.transpose(sequence1, 2, 3)
        transformed2 = torch.transpose(sequence2, 2, 3)
    return transformed1, transformed2


if __name__ == "__main__":
    args = data_loading_parser()
    logger_setup(args)

    dataset = VideoDataset(
        data_dir="data/processed", img_size=(512, 512), rolling_window=5
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(
        f"Dataset sample input shape: {dataset[0][0].shape}, should be (rolling_window, n_channels, img_size[0] // 4, img_size[1] // 4)"
    )
    logging.info(
        f"Dataset sample target shape: {dataset[0][1].shape}, should be (n_channels, img_size[0], img_size[1])"
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    logging.info(
        f"Dataloader sample input shape: {next(iter(dataloader))[0].shape}, should be (batch_size, rolling_window, n_channels, img_size[0] // 4, img_size[1] // 4)"
    )
    logging.info(
        f"Dataloader sample target shape: {next(iter(dataloader))[1].shape}, should be (batch_size, n_channels, img_size[0], img_size[1])"
    )
