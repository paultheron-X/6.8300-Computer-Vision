import os
import sys
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)  # for testing this file with main and so that utils can be imported

import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import glob
import random
from tqdm import tqdm

from utils.logging_config import logger_setup
from utils.arguments_parser import data_loading_parser


class VideoDataset(Dataset):
    def __init__(self, data_dir, is_test=False, img_size=(512, 512), rolling_window=5):
        self.data_dir = data_dir
        self.img_size = img_size
        if rolling_window % 2 == 0:
            raise ValueError("Rolling window must be odd")
        self.rolling_window = rolling_window

        self.total_keys = sorted(os.listdir(data_dir))
        self.total_keys = [i for i in self.total_keys if "downsampled" not in i]

        val_keys = ["031", "042"]  # ['000', '011', '015', '020']
        if is_test:
            self.keys = [i for i in self.total_keys if i in val_keys]
        else:
            self.keys = [i for i in self.total_keys if i not in val_keys]

        self.files_per_key = {
            key: sorted(
                os.listdir(os.path.join(self.data_dir, key))
            )  # get all the files in the key directory
            for key in self.keys
        }
        # process files per key to add 'processed' dir before the file name
        for key in self.keys:
            self.files_per_key[key] = [
                os.path.join(self.data_dir, "downsampled", key, file)
                for file in self.files_per_key[key]
            ]

        self.num_files_per_key = {
            key: len(self.files_per_key[key]) for key in self.keys
        }

    def prepare_data(self):
        """Prepare data for training, this will down sample the images and save them
        Args:
            None
        Returns:
            None
        """
        logging.info("Preparing data")
        for key in tqdm(self.total_keys):
            save_path = os.path.join(self.data_dir, "downsampled", key)
            os.makedirs(save_path, exist_ok=True)

            # now we downsample the images
            images = [
                img_dir
                for img_dir in sorted(os.listdir(os.path.join(self.data_dir, key)))
                if "downsampled" not in img_dir
            ]
            for file in images:
                with Image.open(os.path.join(self.data_dir, key, file)) as img:
                    img = torch.nn.functional.interpolate(
                        TF.ToTensor()(img).unsqueeze(0),
                        scale_factor=1 / 4,
                        mode="bicubic",
                    )
                    # save the downsampled image in the parent directory, with the name downsampled_{original_name}
                    img = img.squeeze(0)
                    img = TF.ToPILImage()(img)
                    img.save(os.path.join(save_path, f"{os.path.basename(file)}"))

    def __len__(self):
        # this dataset will extract batch of rolling_window frames from each video
        # will take the keys sequentially
        return sum(
            [self.num_files_per_key[key] - self.rolling_window + 1 for key in self.keys]
        )

    def __getitem__(self, index):
        # get the key and the index of the file in the key
        key, file_idx = self.get_key_and_file_idx(index)
        # get the file names of the rolling window
        file_names = self.files_per_key[key][
            file_idx
            - (self.rolling_window - 1) // 2 : file_idx
            + (self.rolling_window - 1) // 2
            + 1
        ]
        # gt_image
        gt_image = self.read_image(
            os.path.join(self.data_dir, key, f"frame_{file_idx:04d}.png")
        )
        # read the images
        # TODO: add the transforms
        imgs = [self.read_image(file_name) for file_name in file_names]
        # stack the images
        train_imgs = torch.stack(imgs)  # (t, c, h, w)
        return train_imgs, gt_image

    def get_key_and_file_idx(self, index):
        # get the key and the index of the file in the key
        for key in self.keys:
            if index < self.num_files_per_key[key] - self.rolling_window + 1:
                return (
                    key,
                    index + (self.rolling_window - 1) // 2,
                )  # add the offset to get the index of the middle frame
            else:
                index -= self.num_files_per_key[key] - self.rolling_window + 1
        raise ValueError("Index out of range")

    def read_image(self, path):
        with Image.open(path) as img:
            img = TF.ToTensor()(img)
            img = TF.Resize(self.img_size)(img)
            return img


def generate_segment_indices(
    videopath1, videopath2, num_input_frames=10, filename_tmpl="{:08d}.png"
):
    """generate segment function
    Args:
        videopath1,2 (str): input directory which contains sequential frames
        filename_tmpl (str): template which represents sequential frames
    Returns:
        Tensor, Tensor: Output sequence with shape (t, c, h, w)
    """
    seq_length = len(glob.glob(f"{videopath1}/*.png"))
    seq_length2 = len(glob.glob(f"{videopath2}/*.png"))

    if seq_length != seq_length2:
        raise ValueError(
            f"videopath1 and videopath2 must have same number of frames\nbut they have {seq_length} and {seq_length2}"
        )
    if num_input_frames > seq_length:
        raise ValueError(
            f"num_input_frames{num_input_frames} must be greater than frames in {videopath1} \n and {videopath2}"
        )

    start_frame_idx = np.random.randint(0, seq_length - num_input_frames)
    end_frame_idx = start_frame_idx + num_input_frames
    segment1 = [
        self.read_image(os.path.join(videopath1, filename_tmpl.format(i))) / 255.0
        for i in range(start_frame_idx, end_frame_idx)
    ]
    segment2 = [
        self.read_image(os.path.join(videopath2, filename_tmpl.format(i))) / 255.0
        for i in range(start_frame_idx, end_frame_idx)
    ]
    return torch.stack(segment1), torch.stack(segment2)


def pair_random_crop_seq(hr_seq, lr_seq, patch_size, scale_factor=4):
    """crop image pair for data augment
    Args:
        hr (Tensor): hr images with shape (t, c, 4h, 4w).
        lr (Tensor): lr images with shape (t, c, h, w).
        patch_size (int): the size of cropped image
    Returns:
        Tensor, Tensor: cropped images(hr,lr)
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
