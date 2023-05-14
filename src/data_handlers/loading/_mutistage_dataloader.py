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


from data_handlers.loading._dataloader import VideoDataset


class MultiStageVideoDataset(VideoDataset):

    """
    Returns for the lr: a tuple (input_1, input_2, input_3), with skipping 0, 1 and 2 frames, with the rolling window,
    Truth will be the middle frame (each input will have the middle frame inside).

    Example:
        with rolling window = 5
        input_3 = [0, 3, 6, 9, 12]
        input_2 = [2, 4, 6, 8, 10]
        input_1 = [4, 5, 6, 7, 8]

        + roll that in the whole dataset

        Truth frame is 6 here

    """

    def __init__(
        self,
        lr_data_dir,
        hr_data_dir,
        is_test=False,
        is_val=False,
        rolling_window=5,
        deltas=(1, 2, 3),
        **kwargs
    ):
        super().__init__(
            lr_data_dir, hr_data_dir, is_test, is_val, rolling_window, deltas, **kwargs
        )
        (delta_1, delta_2, delta_3) = (1,3,5) #self.deltas
        ind_mid_frame = self.rolling_window // 2
        # Create group with largest delta (delta_1, delta_2, delta_3)
        self.inds_ind_3 = [i for i in range(0, delta_3 * self.rolling_window, delta_3)]
        self.base_mid_indice = self.inds_ind_3[ind_mid_frame]
        self.inds_ind_2 = [self.base_mid_indice + delta_2 * i for i in range(-ind_mid_frame, ind_mid_frame + 1)]
        self.inds_ind_1 = [self.base_mid_indice + delta_1 * i for i in range(-ind_mid_frame, ind_mid_frame + 1)]
        # self.inds_ind_3 = [i for i in range(0, 3 * self.rolling_window, 3)]
        # ind_mid_frame = self.rolling_window // 2
        # self.base_mid_indice = self.inds_ind_3[ind_mid_frame]

        # self.inds_ind_2 = [
        #     self.base_mid_indice + i
        #     for i in range(-self.rolling_window + 1, self.rolling_window, 2)
        # ]
        # self.inds_ind_1 = [
        #     self.base_mid_indice + i
        #     for i in range(
        #         -self.rolling_window // 2 + 1, self.rolling_window // 2 + 1, 1
        #     )
        # ]

        # we calculate the indices for sliding on the most restrictive: input_3
        indices_slider = {
            key : [i for i in range(0, len(self.files_per_key[key]) - 2 * self.base_mid_indice)]
            for key in self.keys
        }
        
        dict_input_3 = {
            key: [
                [
                    self.files_per_key[key][i + self.inds_ind_3[j]]
                    for j in range(self.rolling_window)
                ]
                for i in indices_slider[key]
            ]
            for key in self.keys
        }

        dict_input_2 = {
            key: [
                [
                    self.files_per_key[key][i + self.inds_ind_2[j]]
                    for j in range(self.rolling_window)
                ]
                    for i in indices_slider[key]
                    ]
            for key in self.keys
        }

        dict_input_1 = {
            key: [
                [
                    self.files_per_key[key][i + self.inds_ind_1[j]]
                    for j in range(self.rolling_window)
                ]
                for i in indices_slider[key]
            ]
            for key in self.keys
        }

        self.frames = {
            "input_3": dict_input_3,
            "input_2": dict_input_2,
            "input_1": dict_input_1,
        }

        self.num_files_per_key = {
            key: len(self.frames["input_1"][key]) for key in self.keys
        }

        self.len = sum(self.num_files_per_key.values())
        
    def __getitem__(self, index):
        # 1. get the key and the index of the file in the key
        # get the key and the index of the file in the key
        key, frame_idx = self.get_key_and_frame_idx(index)
        # get the file names of the rolling window
        file_names_1 = self.frames["input_1"][key][frame_idx]
        file_names_2 = self.frames["input_2"][key][frame_idx]
        file_names_3 = self.frames["input_3"][key][frame_idx]
        
        # 2. get the images
        gt_images = [
            read_image(os.path.join(self.hr_data_dir, key, file_names_1[self.rolling_window // 2])) / 255
        ]
        
        gt_images_tensor = torch.stack(gt_images)
        
        lr_images_1 = [
            read_image(os.path.join(self.lr_data_dir, key, file_name)) / 255
            for file_name in file_names_1
        ]
        
        lr_images_1_tensor = torch.stack(lr_images_1)
        
        lr_images_2 = [
            read_image(os.path.join(self.lr_data_dir, key, file_name)) / 255
            for file_name in file_names_2
        ]
        
        lr_images_2_tensor = torch.stack(lr_images_2)
        
        lr_images_3 = [
            read_image(os.path.join(self.lr_data_dir, key, file_name)) / 255
            for file_name in file_names_3
        ]

        lr_images_3_tensor = torch.stack(lr_images_3)
        
        if not self.is_test and not self.is_val:
            # transform:
            #1. concat all the lr images in the sequence dimension
            lr_images_tensor = torch.cat([lr_images_1_tensor, lr_images_2_tensor, lr_images_3_tensor], dim=0)
            # duplicate the gt image to match the number of lr images
            gt_images_tensor = gt_images_tensor.repeat(lr_images_tensor.shape[0], 1, 1, 1)
            
            #2. apply the transforms
            gt_seq, lr_seq = self.transform(gt_images_tensor, lr_images_tensor)
            
            #3. split the sequence back into 3
            lr_images_1_tensor = lr_seq[:self.rolling_window]
            lr_images_2_tensor = lr_seq[self.rolling_window:2*self.rolling_window]
            lr_images_3_tensor = lr_seq[2*self.rolling_window:]
            
            # keep only the middle frame of the gt sequence
            gt_images_tensor = gt_seq[self.rolling_window]
            
        return (lr_images_1_tensor, lr_images_2_tensor, lr_images_3_tensor), gt_images_tensor
    
if __name__ == "__main__":
    args = data_loading_parser()
    logger_setup(args)

    dataset = MultiStageVideoDataset(
        lr_data_dir="data/processed/train/train_sharp_bicubic/X4",
        hr_data_dir="data/processed/train/train_sharp",
        is_test=False,
        rolling_window=5,
        patch_size=64,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    (in_1, in_2, in_3), target = dataset[0]
    
    logging.info(
        f"Dataset sample input1 shape: {in_1.shape}"
    )
    logging.info(
        f"Dataset sample input2 shape: {in_2.shape}"
    )
    logging.info(
        f"Dataset sample input3 shape: {in_3.shape}"
    )
    logging.info(
        f"Dataset sample target shape: {target.shape}"
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    logging.info(
        f"Dataloader sample input1 shape: {next(iter(dataloader))[0][0].shape}"
    )
    logging.info(
        f"Dataloader sample input2 shape: {next(iter(dataloader))[0][1].shape}"
    )
    logging.info(
        f"Dataloader sample input3 shape: {next(iter(dataloader))[0][2].shape}"
    )
    logging.info(
        f"Dataloader sample target shape: {next(iter(dataloader))[1].shape}"
    )
    
    from time import time
    import multiprocessing as mp
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

            
        

