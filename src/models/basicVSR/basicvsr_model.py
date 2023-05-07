"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

import torch
from torch import nn

from ..optical_flow.SPyNet import SPyNet, get_spynet
from .modules import PixelShuffle, ResidualBlocksWithInputConv, flow_warp

import logging

class basicVSR(nn.Module):
    def __init__(self, scale_factor=4, mid_channels=64, num_blocks=30, spynet_pretrained=None, pretrained_model=None, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.mid_channels = mid_channels

        # alignment(optical flow network)
        #self.optical_module = get_spynet(spynet_pretrained)
        
        # map kwargs to optical flow module
        optical_flow_module = kwargs.get('optical_flow_module', 'SPYNET')
        if optical_flow_module == 'SPYNET':
            self.optical_module = get_spynet(spynet_pretrained)
        elif optical_flow_module == 'RAFT':
            from ..optical_flow import get_raft
            self.optical_module = get_raft(small=False)
        else:
            raise NotImplementedError(f"Optical flow module {kwargs['optical_flow_module']} not implemented")
        
        self.optical_module_name = self.optical_module.__class__.__name__

        # propagation
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShuffle(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShuffle(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if pretrained_model is not None:
            if pretrained_model != "False":
                self.load_pretrained_weights(torch.load(pretrained_model))
                logging.debug(f"Loaded pretrained weights from {pretrained_model}")

        if kwargs.get('reset_spynet', False):
            logging.debug("Resetting SPyNet weights")
            self.optical_module = get_spynet(spynet_pretrained)  # we take the old spynet

    def load_pretrained_weights(self, weights_pret):
        model_keys = [k for k in self.state_dict().keys()]
        pretrained_keys = [k for k in weights_pret.keys()]

        new_dict = {}
        if self.optical_module_name == "SPyNet":
            for i, a in enumerate(model_keys):
                new_dict[a] = weights_pret[a] # all the same
        else:
            # the optical flow network is raft: load the pretrained weights for the whole model,
            # except for the optical flow network where we keep raft
            for i, a in enumerate(model_keys):
                if "optical_module" in a:
                    new_dict[a] = self.state_dict()[a] # keep the raft weights
                else:
                    new_dict[a] = weights_pret[a] # all the same

        self.load_state_dict(new_dict, strict=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        
        name = self.optical_module_name
        
        if name == "SPyNet":
            flows_backward = self.optical_module(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        else:
            flows_backward = self.optical_module(lrs_1, lrs_2)[-1].view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            if name == "SPyNet":
                flows_forward = self.optical_module(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
            else:
                flows_forward = self.optical_module(lrs_2, lrs_1)[-1].view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward
    
    def forward_core(self, lrs):
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, "The height and width of inputs should be at least 64, " f"but got {h} and {w}."

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs_backward = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs_backward.append(feat_prop)
        outputs_backward = outputs_backward[::-1]

        # forward-time propagation
        outputs_forward = []
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            outputs_forward.append(feat_prop)

        return outputs_forward, outputs_backward
    
    def forward_fusion(self, outputs_forward_i, outputs_backward_i):
        out = torch.cat([outputs_backward_i, outputs_forward_i], dim=1)
        out = self.lrelu(self.fusion(out))

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        
        outputs_forward, outputs_backward = self.forward_core(lrs)
        
        # upsampling
        outputs = []
        for i in range(t):
            out = self.forward_fusion(outputs_forward[i], outputs_backward[i])
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lrs[:, i, :, :, :])
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)
    
class BasicVSRCore(basicVSR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, lrs):
        return self.forward_core(lrs)
    
class BasicVSRFusion(basicVSR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        
        outputs_forward, outputs_backward = self.forward_core(lrs)
        
        # upsampling
        outputs = []
        for i in range(t):
            out = self.forward_fusion(outputs_forward[i], outputs_backward[i])
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)

