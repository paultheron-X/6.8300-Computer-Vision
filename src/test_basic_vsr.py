import logging
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

from utils.logging_config import logger_setup
from utils.arguments_parser import args_parser
from config import return_config

import os
import argparse
from typing import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_handlers.loading import VideoDataset
from models import basicVSR
from utils.loss import CharbonnierLoss
from utils.utils_general import resize_sequences


def main(config):
    # set the seeds
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting main training script")

    logging.info("Loading test data")
    #logging.debug(f"Creating dataset from path: {config['data_path']}")

    
    test_dataset = VideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=True,
    )

    logging.debug(f"Creating train and test dataloaders")
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = basicVSR(spynet_pretrained=config["spynet_pretrained"], pretrained_model=config["basic_vsr_pretrained"]).to(device)
    criterion_mse = nn.MSELoss().to(device)
    max_epoch = 1

    os.makedirs(f'{config["log_dir"]}/models', exist_ok=True)
    os.makedirs(f'{config["log_dir"]}/images', exist_ok=True)

    logging.info("Starting testing")
    for epoch in range(max_epoch):
        logging.debug(f"Starting validation at epoch {epoch+1}")
        model.eval()
        val_psnr, lq_psnr = 0, 0
        os.makedirs(f'{config["log_dir"]}/images/epoch{epoch+1:05}', exist_ok=True)
        with torch.no_grad():
            with tqdm(test_loader, ncols=100) as pbar:
                for idx, data in enumerate(pbar):
                    gt_sequences, lq_sequences = Variable(data[1]), Variable(data[0])
                    gt_sequences = gt_sequences.to(device)
                    lq_sequences = lq_sequences.to(device)
                    pred_sequences = model(lq_sequences)
                    lq_mid = resize_sequences(
                        lq_sequences, (gt_sequences.size(dim=2), gt_sequences.size(dim=3))
                    )
                    mid_frame = config["rolling_window"] // 2
                    pred_sequences = pred_sequences[:,mid_frame,:,:,:]
                    
                    lq_mid = lq_mid[:,mid_frame,:,:,:]
                    
                    
                    val_mse = criterion_mse(pred_sequences, gt_sequences)
                    lq_mse = criterion_mse(lq_mid, gt_sequences)
                    val_psnr += 10 * log10(1 / val_mse.data)
                    lq_psnr += 10 * log10(1 / lq_mse.data)
                    pbar.set_description(
                        f"PSNR:{val_psnr / (idx + 1):.2f},(lq:{lq_psnr/(idx + 1):.2f})"
                    )
                    
                    save_image(
                        pred_sequences[0],
                        f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_SR.png',
                        nrow=5,
                    )
                    save_image(
                        lq_mid[0],
                        f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_LQ.png',
                        nrow=5,
                    )
                    save_image(
                        gt_sequences[0],
                        f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_GT.png',
                        nrow=5,
                    )

        logging.info(
            f"==[validation]== PSNR:{val_psnr / len(test_loader):.2f},(lq:{lq_psnr/len(test_loader):.2f})"
        )
        torch.save(model.state_dict(), f'{config["log_dir"]}/models/model_{epoch}.pth')


if __name__ == "__main__":
    args = args_parser()

    # TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    # ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)

    logger_setup(args)

    config.update(vars(args))

    main(config)
