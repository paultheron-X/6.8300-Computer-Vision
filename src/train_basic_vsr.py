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

    logging.info("Loading data")
    logging.debug(f"Creating dataset from path: {config['data_path']}")

    train_dataset = VideoDataset(
        data_dir=config["data_path"], rolling_window=config["rolling_window"]
    )
    test_dataset = VideoDataset(
        data_dir=config["data_path"],
        rolling_window=config["rolling_window"],
        is_test=True,
    )

    # if config prepare data is true, prepare the data, or if the data is not prepared
    if config["prepare_data"] or not os.path.exists(os.path.join(config["data_path"], "downsampled")):
        train_dataset.prepare_data()  # will prepare for all

    logging.debug(f"Creating train and test dataloaders")
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = basicVSR(spynet_pretrained=config["spynet_pretrained"]).to(device)

    criterion = CharbonnierLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.spynet.parameters(), "lr": 2.5e-5},
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

    max_epoch = config["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-7)

    os.makedirs(f'{config["log_dir"]}/models', exist_ok=True)
    os.makedirs(f'{config["log_dir"]}/images', exist_ok=True)

    logging.info("Starting training")
    train_loss = []
    validation_loss = []
    for epoch in range(max_epoch):
        model.train()
        # fix SPyNet and EDVR at first 5000 iteration
        if epoch < 5000:
            for k, v in model.named_parameters():
                if "spynet" in k or "edvr" in k:
                    v.requires_grad_(False)
        elif epoch == 5000:
            # train all the parameters
            model.requires_grad_(True)

        epoch_loss = 0
        with tqdm(train_loader, ncols=100) as pbar:
            for idx, data in enumerate(pbar):
                gt_sequences, lq_sequences = Variable(data[1]), Variable(data[0])
                gt_sequences = gt_sequences.to(device)
                lq_sequences = lq_sequences.to(device)

                pred_sequences = model(lq_sequences)
                loss = criterion(pred_sequences, gt_sequences)
                epoch_loss += loss.item()
                # epoch_psnr += 10 * log10(1 / loss.data)

                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_description(f"[Epoch {epoch+1}]")
                pbar.set_postfix(OrderedDict(loss=f"{loss.data:.3f}"))

            train_loss.append(epoch_loss / len(train_loader))

        if (epoch + 1) % config["val_interval"] != 0:
            continue

        logging.debug(f"Starting validation at epoch {epoch+1}")
        model.eval()
        val_psnr, lq_psnr = 0, 0
        os.makedirs(f'{config["log_dir"]}/images/epoch{epoch+1:05}', exist_ok=True)
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                gt_sequences, lq_sequences = data
                gt_sequences = gt_sequences.to(device)
                lq_sequences = lq_sequences.to(device)
                pred_sequences = model(lq_sequences)
                lq_sequences = resize_sequences(
                    lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4))
                )
                val_mse = criterion_mse(pred_sequences, gt_sequences)
                lq_mse = criterion_mse(lq_sequences, gt_sequences)
                val_psnr += 10 * log10(1 / val_mse.data)
                lq_psnr += 10 * log10(1 / lq_mse.data)

                save_image(
                    pred_sequences[0],
                    f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_SR.png',
                    nrow=5,
                )
                save_image(
                    lq_sequences[0],
                    f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_LQ.png',
                    nrow=5,
                )
                save_image(
                    gt_sequences[0],
                    f'{config["log_dir"]}/images/epoch{epoch+1:05}/{idx}_GT.png',
                    nrow=5,
                )

            validation_loss.append(epoch_loss / len(test_loader))

        logging.info(
            f"==[validation]== PSNR:{val_psnr / len(test_loader):.2f},(lq:{lq_psnr/len(test_loader):.2f})"
        )
        torch.save(model.state_dict(), f'{config["log_dir"]}/models/model_{epoch}.pth')

    fig = plt.figure()
    train_loss = [loss for loss in train_loss]
    validation_loss = [loss for loss in validation_loss]
    x_train = list(range(len(train_loss)))
    x_val = [x for x in range(max_epoch) if (x + 1) % config["val_interval"] == 0]
    plt.plot(x_train, train_loss)
    plt.plot(x_val, validation_loss)

    fig.savefig(f'{config["log_dir"]}/loss.png')


if __name__ == "__main__":
    args = args_parser()

    # TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    # ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)

    logger_setup(args)

    config.update(vars(args))

    main(config)
