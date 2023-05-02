import torch
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image

from utils.utils_general import resize_sequences
from typing import OrderedDict

from torch.cuda.amp import autocast


def train_loop(
    model,
    epoch,
    config,
    device,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    grad_accumulation_steps=1,
):
    epoch_loss = 0
    with tqdm(train_loader, ncols=100) as pbar:
        for idx, data in enumerate(pbar):
            optimizer.zero_grad()
            gt_sequences, lq_sequences = Variable(data[0]), Variable(data[1])

            gt_sequences = gt_sequences.to(device)
            lq_sequences = lq_sequences.to(device)

            with autocast():
                pred_sequences = model(lq_sequences)
                mid_frame = config["rolling_window"] // 2
                pred_sequences = pred_sequences[
                    :, mid_frame, :, :, :
                ]  # TODO challenge that: shuld e compute the loss on all the reconstructed frames ??
                gt_sequences = gt_sequences[:, mid_frame, :, :, :]

                loss = criterion(pred_sequences, gt_sequences)
                epoch_loss += loss.item()
            # epoch_psnr += 10 * log10(1 / loss.data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_description(f"[Epoch {epoch+1}]")
            pbar.set_postfix(OrderedDict(loss=f"{loss.data:.3f}"))
