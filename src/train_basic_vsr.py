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

from utils.tester import test_loop
from utils.trainer import train_loop

from torch.cuda.amp import GradScaler


def main(config):
    # set the seeds
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting main training script")

    logging.info("Loading data")
    logging.debug(f"Creating dataset from path: {config['lr_data_dir']}")

    train_dataset = VideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        patch_size=config["patch_size"],
        skip_frames=config["skip_frames"],
    )
    test_dataset = VideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=True,
        is_small_test=False,
        patch_size=config["patch_size"],
        skip_frames=config["skip_frames"],
    )

    val_dataset = VideoDataset(
        lr_data_dir=config["lr_data_dir"],
        hr_data_dir=config["hr_data_dir"],
        rolling_window=config["rolling_window"],
        is_test=True,
        is_small_test=True,
        patch_size=config["patch_size"],
        skip_frames=config["skip_frames"],
    )

    logging.debug(f"Creating train and test dataloaders")
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = basicVSR(
        spynet_pretrained=config["spynet_pretrained"],
        pretrained_model=config["basic_vsr_pretrained"],
    ).to(device)

    criterion = CharbonnierLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.optical_module.parameters(), "lr": 2.5e-5},
            {"params": model.backward_resblocks.parameters()},
            {"params": model.forward_resblocks.parameters()},
            {"params": model.fusion.parameters()},
            {"params": model.upsample1.parameters()},
            {"params": model.upsample2.parameters()},
            {"params": model.conv_hr.parameters()},
            {"params": model.conv_last.parameters()},
        ],
        lr=2e-4,
        betas=(0.9, 0.99),
    )
    scaler = GradScaler()

    max_epoch = config["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-7)

    os.makedirs(f'{config["result_dir"]}/models', exist_ok=True)
    os.makedirs(f'{config["result_dir"]}/images', exist_ok=True)

    logging.info("Starting training")
    train_loss = []
    validation_loss = []

    comp_model = torch.compile(model, backend="aot_eager")
    for epoch in range(max_epoch):
        comp_model.train()
        # fix SPyNet and EDVR at first 5000 iteration
        if epoch < 5000:
            for k, v in comp_model.named_parameters():
                if "spynet" in k or "edvr" in k:
                    v.requires_grad_(False)
        elif epoch == 5000:
            # train all the parameters
            comp_model.requires_grad_(True)

        epoch_loss = train_loop(
            comp_model,
            epoch,
            config,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
        )

        train_loss.append(epoch_loss / len(train_loader))
        #if (epoch + 1) % config["val_interval"] != 0:
        #    continue

        logging.debug(f"Starting validation at epoch {epoch+1}")

        # with val loader it is fast eval
        val_loss = test_loop(
            comp_model, max_epoch, config, device, val_loader, criterion_mse
        )

        if epoch % 10 == 0 and epoch != 0:
            # we run a full eval on the test set
            _ = test_loop(
                comp_model, max_epoch, config, device, test_loader, criterion_mse
            )

        validation_loss.append(val_loss / len(val_loader))

    fig = plt.figure()
    train_loss = [loss for loss in train_loss]
    validation_loss = [loss for loss in validation_loss]
    x_train = list(range(len(train_loss)))
    x_val = [x for x in range(max_epoch) if (x + 1) % config["val_interval"] == 0]
    plt.plot(x_train, train_loss)
    plt.plot(x_val, validation_loss)

    fig.savefig(f'{config["result_dir"]}/loss.png')


if __name__ == "__main__":
    args = args_parser()

    # TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    # ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)

    logger_setup(args)

    config.update(vars(args))

    main(config)
