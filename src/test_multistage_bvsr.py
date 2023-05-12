import logging
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

from utils.logging_config import logger_setup
from utils.arguments_parser import args_parser
from config import return_config

import os
import matplotlib.pyplot as plt


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_handlers.loading import MultiStageVideoDataset
from models import MultiStageBasicVSR
from utils.loss import CharbonnierLoss

from utils.tester_multistage import test_loop
from utils.trainer_multistage import train_loop

from torch.cuda.amp import GradScaler


def main(config):
    # set the seeds
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting test script for multistage bvsr")

    logging.info("Loading data")
    logging.debug(f"Creating dataset from path: {config['lr_data_dir']}")

    test_dataset = MultiStageVideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=True,
        is_val=False,
        patch_size=config["patch_size"],
        skip_frames=config["skip_frames"],
    )

    val_dataset = MultiStageVideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=False,
        is_val=True,
        patch_size=config["patch_size"],
        skip_frames=config["skip_frames"],
    )

    logging.debug(f"Creating val and test dataloaders")
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4)

    model = MultiStageBasicVSR(
        spynet_pretrained=config["spynet_pretrained"],
        pretrained_bvsr=config["basic_vsr_pretrained"],
        pretrained_model=config["mstage_vsr_pretrained"],
        rolling_window=config["rolling_window"],
    ).to(device)

    criterion_mse = nn.MSELoss().to(device)

    os.makedirs(f'{config["result_dir"]}/models', exist_ok=True)
    os.makedirs(f'{config["result_dir"]}/images', exist_ok=True)

    logging.info("Starting testing")

    comp_model = torch.compile(model, backend="aot_eager")

    # with val loader it is fast eval
    logging.info("Starting validationset test")
    _ = test_loop(comp_model, 0, config, device, val_loader, criterion_mse)

    logging.info("Starting testset test")
    _ = test_loop(comp_model, 1, config, device, test_loader, criterion_mse)


if __name__ == "__main__":
    args = args_parser()

    # TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    # ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)

    logger_setup(args)

    config.update(vars(args))

    main(config)
