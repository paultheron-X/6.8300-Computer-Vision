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

import math

import numpy as np
import torch

from cv2 import resize as cv2_resize

from utils.logging_config import logger_setup
from utils.arguments_parser import data_loading_parser

class RandomDownSampling:
    """Generate LQ image from GT (and crop), which will randomly pick a scale.
    Args:
        scale_min (float): The minimum of upsampling scale, inclusive.
            Default: 1.0.
        scale_max (float): The maximum of upsampling scale, exclusive.
            Default: 4.0.
        patch_size (int): The cropped lr patch size.
            Default: None, means no crop.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            Default: "bicubic".
        Scale will be picked in the range of [scale_min, scale_max).
    """

    def __init__(self,
                 scale_min=1.0,
                 scale_max=4.0,
                 patch_size=64,
                 interpolation='bicubic',
        ):
        assert scale_max >= scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.backend = 'cv2'

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. 'gt' is required.
        Returns:
            dict: A dict containing the processed data and information.
                modified 'gt', supplement 'lq' and 'scale' to keys.
        """
        img = results['gt']
        scale = np.random.uniform(self.scale_min, self.scale_max)

        if self.patch_size is None:
            h_lr = math.floor(img.shape[-3] / scale + 1e-9)
            w_lr = math.floor(img.shape[-2] / scale + 1e-9)
            img = img[:round(h_lr * scale), :round(w_lr * scale), :]
            img_down = resize_fn(img, (w_lr, h_lr), self.interpolation,
                                 self.backend)
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.patch_size
            w_hr = round(w_lr * scale)
            x0 = np.random.randint(0, img.shape[-3] - w_hr)
            y0 = np.random.randint(0, img.shape[-2] - w_hr)
            crop_hr = img[x0:x0 + w_hr, y0:y0 + w_hr, :]
            crop_lr = resize_fn(crop_hr, w_lr, self.interpolation,
                                self.backend)
        results['gt'] = crop_hr
        results['lq'] = crop_lr
        results['scale'] = scale

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f' scale_min={self.scale_min}, '
                     f'scale_max={self.scale_max}, '
                     f'patch_size={self.patch_size}, '
                     f'interpolation={self.interpolation}, '
                     f'backend={self.backend}')

        return repr_str


def resize_fn(img, size, interpolation='bicubic', backend='cv2'):
    """Resize the given image to a given size.
    Args:
        img (ndarray | torch.Tensor): The input image.
        size (int | tuple[int]): Target size w or (w, h).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear", "bicubic", "box", "lanczos",
            "hamming" for 'pillow' backend.
            Default: "bicubic".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used.
            Default: "pillow".
    Returns:
        ndarray | torch.Tensor: `resized_img`, whose type is same as `img`.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        return imresize(
            img, size, interpolation=interpolation, backend=backend)
    elif isinstance(img, torch.Tensor):
        image = imresize(
            img.numpy(), size, interpolation=interpolation, backend=backend)
        return torch.from_numpy(image)

    else:
        raise TypeError('img should got np.ndarray or torch.Tensor,'
                        f'but got {type(img)}')
        
        
def imresize(img, size, interpolation='bicubic', backend='pillow'):
    """Resize the given image to a given size.
    Args:
        img (ndarray): The input image.
        size (int | tuple[int]): Target size w or (w, h).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear", "bicubic", "box", "lanczos",
            "hamming" for 'pillow' backend.
            Default: "bicubic".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used.
            Default: "pillow".
    Returns:
        ndarray: `resized_img`.
    """
    if backend == 'cv2':
        return cv2_resize(img, size, interpolation=interpolation)
    else:
        raise ValueError(f'backend {backend} is not supported')



if __name__ == "__main__":
    args = data_loading_parser()
    logger_setup(args)

    # create dataset
    downsampler = RandomDownSampling()
    
    logging.info("Loading raw data...")
    
    
    raw_data_dir = 'data/raw/train_orig'
    savind_dir = 'data/REDS/bicubic'
    
    logging.debug('Loading raw data from: {}'.format(raw_data_dir))
    logging.debug('Current working directory: {}'.format(os.getcwd()))
    raw_data = os.listdir(raw_data_dir) # all the videos numbers
    
    os.makedirs(savind_dir, exist_ok=True)
    
    for video in raw_data:
        os.makedirs(os.path.join(savind_dir, video), exist_ok=True)
        logging.debug('Processing video: {}'.format(video))
        video_path = os.path.join(raw_data_dir, video)
        video_frames = os.listdir(video_path)
        video_frames.sort()
        for frame in video_frames:
            frame_path = os.path.join(video_path, frame)
            frame = cv2.imread(frame_path)
            frame = downsampler({'gt': frame})
            cv2.imwrite(os.path.join(savind_dir, video, frame), frame)
        
    
    
    
    
