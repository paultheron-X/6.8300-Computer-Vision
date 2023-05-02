import logging
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

from utils.logging_config import logger_setup
from utils.arguments_parser import args_parser
from config import return_config

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_handlers.loading import VideoDataset
from models import basicVSR

from utils.tester import test_loop


def main(config):
    # set the seeds
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting main training script")

    logging.info("Loading test data")

    test_dataset = VideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=True,
        skip_frames=config["skip_frames"],
    )

    logging.debug(f"Creating train and test dataloaders")

    if config["rolling_window"] == 25:
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    else:
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = basicVSR(
        spynet_pretrained=config["spynet_pretrained"],
        pretrained_model=config["basic_vsr_pretrained"],
        reset_spynet=config["reset_spynet"],
        optical_flow_module=config["optical_flow_module"],
    ).to(device)
    criterion_mse = nn.MSELoss().to(device)
    max_epoch = 1

    os.makedirs(f'{config["result_dir"]}/models', exist_ok=True)
    os.makedirs(f'{config["result_dir"]}/images', exist_ok=True)

    logging.info("Starting testing")

    model = test_loop(model, max_epoch, config, device, test_loader, criterion_mse)


if __name__ == "__main__":
    args = args_parser()

    # TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    # ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)

    logger_setup(args)

    config.update(vars(args))

    main(config)
