import torch
import logging
import os
from tqdm import tqdm

import torch
from torch.autograd import Variable

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
    loss_curve = []
    grad_accumulation_steps = config.get("grad_accum_steps", 1)
    skip_frames = config.get("skip_frames", 0)
    if skip_frames:
        mid_frame = config["rolling_window"] // 2
    loss = torch.zeros(1, device=device)
    with tqdm(train_loader, ncols=100) as pbar:
        pbar.set_description(f"[Epoch {epoch+1}]")
        for idx, data in enumerate(pbar):
            optimizer.zero_grad()
            gt_sequences, lq_sequences = data[1], data[0]

            gt_sequences = gt_sequences.to(device)
            
            (in_1, in_2, in_3) = (lq_sequences[0].to(device), lq_sequences[1].to(device), lq_sequences[2].to(device))

            with autocast():
                pred_sequences = model((in_1, in_2, in_3))

                loss_batch = criterion(pred_sequences, gt_sequences)
                epoch_loss += loss_batch.item()
            # epoch_psnr += 10 * log10(1 / loss.data)
            loss += loss_batch
            if (idx + 1) % grad_accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss = torch.zeros(1, device=device)
            
            if idx % 10 == 0:
                loss_curve.append(loss_batch.item())

            pbar.set_postfix(OrderedDict(loss=f"{loss_batch.data:.10f}"))

    return epoch_loss, loss_curve
