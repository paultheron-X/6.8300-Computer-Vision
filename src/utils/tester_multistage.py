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

from torch.cuda.amp import autocast



def test_loop(model, epoch, config, device, test_loader, criterion_mse):
    model.eval()
    val_psnr, lq_psnr = 0, 0
    os.makedirs(f'{config["result_dir"]}/images/epoch{epoch+1:05}', exist_ok=True)
    with torch.no_grad():
        with tqdm(test_loader, ncols=100) as pbar:
            for idx, data in enumerate(pbar):
                gt_sequences, lq_sequences = data[1], data[0]

                gt_sequences = gt_sequences.to(device)
                gt_sequences = gt_sequences.squeeze(1)
            
                (in_1, in_2, in_3) = (lq_sequences[0].to(device), lq_sequences[1].to(device), lq_sequences[2].to(device))
                
                with autocast():
                    pred_sequences = model((in_1, in_2, in_3))

                # compute the loss only on the middle frame of the rolling window
                mid_frame = in_1.shape[1] // 2
                lq_mid = in_1[:, mid_frame, :, :, :]
                lq_mid = resize_sequences(lq_mid.unsqueeze(1), pred_sequences.shape[2:]).squeeze(1)

                val_mse = criterion_mse(pred_sequences, gt_sequences)
                lq_mse = criterion_mse(lq_mid, gt_sequences)
                val_psnr += 10 * log10(1 / val_mse.data)
                lq_psnr += 10 * log10(1 / lq_mse.data)
                pbar.set_description(
                    f"PSNR:{val_psnr / (idx + 1):.2f},(lq:{lq_psnr/(idx + 1):.2f})"
                )

                save_image(
                    pred_sequences[0],
                    f'{config["result_dir"]}/images/epoch{epoch+1:05}/{idx}_SR.png',
                    nrow=5,
                )
                save_image(
                    lq_mid[0],
                    f'{config["result_dir"]}/images/epoch{epoch+1:05}/{idx}_LQ.png',
                    nrow=5,
                )
                save_image(
                    gt_sequences[0],
                    f'{config["result_dir"]}/images/epoch{epoch+1:05}/{idx}_GT.png',
                    nrow=5,
                )
                val_loss = val_mse.item()

        logging.info(
            f"==[validation]== PSNR:{val_psnr / len(test_loader):.2f},(lq:{lq_psnr/len(test_loader):.2f})"
        )
        # TODO: Implement checkpoint saving (not at every epoch but keep the best one only)
        torch.save(
            model.state_dict(), f'{config["result_dir"]}/models/model_{epoch}.pth'
        )
        # write the psnr to a file
        with open(f'{config["result_dir"]}/psnr.txt', "a") as f:
            f.write(f"{epoch} {val_psnr / len(test_loader)}\n")
    return val_loss
