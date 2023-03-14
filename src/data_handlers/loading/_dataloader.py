import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #for testing this file with main and so that utils can be imported

import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

from utils.logging_config import logger_setup
from utils.arguments_parser import data_loading_parser


class VideoDataset(Dataset):
    def __init__(self, data_dir, img_size, rolling_window):
        self.data_dir = data_dir
        self.img_size = img_size
        self.rolling_window = rolling_window
        self.downsample = TF.Compose(
            [TF.Resize((img_size[0] // 4, img_size[1] // 4)), TF.ToTensor()]
        )
        self.video_files = sorted(os.listdir(data_dir))

        self.files = []
        for video_file in self.video_files:
            video_path = os.path.join(self.data_dir, video_file)
            video_files = sorted(os.listdir(video_path))
            for video_file in video_files:
                self.files.append(os.path.join(video_path, video_file))

    def __len__(self):
        return len(self.files) - (self.rolling_window - 1)

    def __getitem__(self, index):
        imgs = []
        for i in range(index, index + self.rolling_window):
            img_path = self.files[i]
            with Image.open(img_path) as img:
                if i - index == self.rolling_window//2:
                    # we save the target image before downsampling
                    target = TF.ToTensor()(img)
                img = self.downsample(img)
                imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs, target


if __name__ == "__main__":
    args = data_loading_parser()
    logger_setup(args)
    
    
    dataset = VideoDataset(
        data_dir="data/processed", img_size=(512, 512), rolling_window=5
    )
    
    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Dataset sample input shape: {dataset[0][0].shape}, should be (rolling_window, n_channels, img_size[0] // 4, img_size[1] // 4)")
    logging.info(f"Dataset sample target shape: {dataset[0][1].shape}, should be (n_channels, img_size[0], img_size[1])")


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    logging.info(f"Dataloader sample input shape: {next(iter(dataloader))[0].shape}, should be (batch_size, rolling_window, n_channels, img_size[0] // 4, img_size[1] // 4)")
    logging.info(f"Dataloader sample target shape: {next(iter(dataloader))[1].shape}, should be (batch_size, n_channels, img_size[0], img_size[1])")
    
