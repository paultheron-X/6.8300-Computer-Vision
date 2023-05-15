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
from .metrics import calculate_ssim
from torch.cuda.amp import autocast


def test_loop(model, epoch, config, device, test_loader, criterion_mse):
    model.eval()
    val_psnr, lq_psnr = 0, 0
    ssim = 0
    os.makedirs(f'{config["result_dir"]}/images/epoch{epoch+1:05}', exist_ok=True)
    with torch.no_grad():
        with tqdm(test_loader, ncols=100) as pbar:
            for idx, data in enumerate(pbar):
                gt_sequences, lq_sequences = Variable(data[1]), Variable(data[0])
                gt_sequences = gt_sequences.to(device)
                lq_sequences = lq_sequences.to(device)
                with autocast():
                    pred_sequences = model(lq_sequences)
                lq_mid = resize_sequences(lq_sequences, pred_sequences.shape[-2:])

                # compute the loss only on the middle frame of the rolling window
                mid_frame = pred_sequences.shape[1] // 2
                #pred_sequences = pred_sequences[:, mid_frame, :, :, :]
                #gt_sequences = gt_sequences[:, mid_frame, :, :, :]
                #lq_mid = lq_mid[:, mid_frame, :, :, :]
                
                #print(pred_sequences.shape, gt_sequences.shape, lq_mid.shape)
                
                val_mse = criterion_mse(pred_sequences, gt_sequences)
                lq_mse = criterion_mse(lq_mid, gt_sequences)
                val_psnr += 10 * log10(1 / val_mse.data)
                lq_psnr += 10 * log10(1 / lq_mse.data)
                
                bs = pred_sequences.shape[0]
                pred_np = pred_sequences.detach().cpu().numpy()
                gt_np = gt_sequences.detach().cpu().numpy()
                #ssim_b = 0
                #print(bs)
                #for i in range(bs):
                #    #print(pred_np[i].shape, gt_np[i].shape)
                #    ssim_b += calculate_ssim(pred_np[i]*255, gt_np[i]*255, crop_border=0, input_order='CHW')
                #    #print(ssim)
                #ssim_b  = ssim_b/bs
                #ssim += ssim_b
                
                
                pbar.set_description(
                    f"PSNR:{val_psnr / (idx + 1):.2f},(lq:{lq_psnr/(idx + 1):.2f}) | SSIM:{ssim / (idx + 1):.2f}"
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
            f"==[validation]== PSNR:{val_psnr / len(test_loader):.2f},(lq:{lq_psnr/len(test_loader):.2f}) | SSIM:{ssim / len(test_loader):.2f}"
        )
        # TODO: Implement checkpoint saving (not at every epoch but keep the best one only)
        torch.save(
            model.state_dict(), f'{config["result_dir"]}/models/model_{epoch}.pth'
        )
        # write the psnr to a file
        with open(f'{config["result_dir"]}/psnr.txt', "a") as f:
            f.write(f"EPOCH: {epoch} | PSNR: {val_psnr / len(test_loader)}\n | SSIM: {ssim / len(test_loader)}\n")
    return val_loss
