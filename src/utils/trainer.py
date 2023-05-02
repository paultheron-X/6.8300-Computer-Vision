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
):
    epoch_loss = 0
    grad_accumulation_steps = config.get("grad_accum_steps", 1)
    skip_frames = config.get("skip_frames", 0)
    if skip_frames:
        mid_frame = config["rolling_window"] // 2
    loss = torch.zeros(1, device=device)
    with tqdm(train_loader, ncols=100) as pbar:
        pbar.set_description(f"[Epoch {epoch+1}]")
        for idx, data in enumerate(pbar):
            optimizer.zero_grad()
            gt_sequences, lq_sequences = Variable(data[0]), Variable(data[1])

            gt_sequences = gt_sequences.to(device)
            lq_sequences = lq_sequences.to(device)

            with autocast():
                pred_sequences = model(lq_sequences)
                
                if skip_frames:
                    pred_sequences = pred_sequences[
                        :, mid_frame, :, :, :
                    ]
                    gt_sequences = gt_sequences[:, mid_frame, :, :, :]
                else:
                    # keep all the frames except the first and last
                    pred_sequences = pred_sequences[:, 1:-1, :, :, :]
                    gt_sequences = gt_sequences[:, 1:-1, :, :, :]

                loss_batch = criterion(pred_sequences, gt_sequences)
                epoch_loss += loss_batch.item()
            # epoch_psnr += 10 * log10(1 / loss.data)
            loss +=loss_batch
            if (idx + 1) % grad_accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss = torch.zeros(1, device=device)

            pbar.set_postfix(OrderedDict(loss=f"{loss_batch.data:.10f}"))
