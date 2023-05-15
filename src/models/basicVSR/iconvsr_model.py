"""
This code is based on Open-MMLab's one.
https://github.com/open-mmlab/mmediting
"""

import torch
from torch import nn
import torch.nn.functional as F 

from ..optical_flow.SPyNet import SPyNet, get_spynet
from .modules import PixelShuffle, ResidualBlocksWithInputConv, flow_warp

import logging

class iconVSR(nn.Module):
    def __init__(self, scale_factor=4, mid_channels=64, num_blocks=30, spynet_pretrained=None, pretrained_model=None,
                 keyframe_stride=5, temporal_padding=2, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.mid_channels = mid_channels
        
        self.t_pad = temporal_padding
        self.kframe_stride = keyframe_stride

        self.edvr = EDVRExtractor(num_frame=temporal_padding*2 + 1,
                                      center_frame_idx=temporal_padding)
        
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
        
        # info refill
        self.backward_fuse = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.forward_fuse = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
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
    
    def extract_refill_features(self, lrs, keyframe_idx):
        """Compute the features for information refill.

        We use EDVR-M to extract features from the selected keyframes
        and its neighbor. The window size in EDVR-M is 5 for REDS and
        7 for Vimeo-90K (following the settings in EDVR).

        Args:
            lrs (Tensor): The input LR sequence with shape (n, t, c, h, w).
            keyframe_idx (list[int]): List of the indices of the selected
                keyframes.

        Returns:
            dict: The features for information-refill. The keys are the
                corresponding index.

        """
        lrs_start = lrs[:, 1+self.t_pad : 1+self.t_pad*2].flip(1)
        lrs_end = lrs[:, -1-self.t_pad*2 : -1-self.t_pad].flip(1)
        lrs = torch.cat([lrs_start, lrs, lrs_end], dim=1)
        num_frame = 2 * self.t_pad + 1

        refill_feat = {}
        for i in keyframe_idx:
            refill_feat[i] = self.edvr(lrs[:, i:i + num_frame].contiguous())
        return refill_feat
    
    def spatial_padding(self, lrs):
        """ Apply spatial pdding.

        Since the PCD module in EDVR requires a resolution of a multiple of 4, 
        we use reflect padding on the LR frame to match the requirements..

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).

        """
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)
    
    def forward_core(self, lrs):
        n, t, c, h_in, w_in = lrs.size()
        assert h_in >= 64 and w_in >= 64, "The height and width of inputs should be at least 64, " f"but got {h} and {w}."

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)
        
        # padding
        lrs = self.spatial_padding(lrs)
        h, w = lrs.size(3), lrs.size(4)

        # get the keyframe for information-refill
        keyframe_idx = list(range(0, t, self.kframe_stride))
        if keyframe_idx[-1] != t-1:
            keyframe_idx.append(t-1) # the last frame is a keyframe
        
        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        refill_feat = self.extract_refill_features(lrs, keyframe_idx)

        # backward-time propagation
        outputs_backward = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, ]
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, refill_feat[i]], dim=1)
                feat_prop = self.backward_fuse(feat_prop)
            feat_prop = torch.cat([feat_prop, curr_lr], dim=1) # no copy?
            feat_prop = self.backward_resblocks(feat_prop)

            outputs_backward.append(feat_prop)
        outputs_backward = outputs_backward[::-1]

        # forward-time propagation
        outputs_forward = []
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, refill_feat[i]], dim=1)
                feat_prop = self.forward_fuse(feat_prop)
            feat_prop = torch.cat([curr_lr, outputs_backward[i], feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            outputs_forward.append(feat_prop)

        return outputs_forward, outputs_backward
    
    def forward_fusion(self, outputs_forward_i, outputs_backward_i):
        out = torch.cat([outputs_backward_i, outputs_forward_i], dim=1)
        out = self.lrelu(self.fusion(out))
        return out

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
    
class EDVRExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.

    We use EDVR-M in IconVSR.

    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5,
                deformable_groups=8, num_extract_block=5,
                center_frame_idx=None, hr_in=None, 
                with_predeblur=False, with_tsa=True):
        super(EDVRExtractor, self).__init__()

        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.pre_deblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        
        # extract pyramid features 
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        
        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        n, t, c, h, w = x.size()

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')
        
        # extract features for each frame
        # Level 1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.pre_deblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        
        feat_l1 = self.feature_extraction(feat_l1)

        # Level 2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))

        # Level 3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(n, t, -1, h, w)
        feat_l2 = feat_l2.view(n, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(n, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(n, -1, h, w)
        feat = self.fusion(aligned_feat)

        return feat


class iconVSRCore(iconVSR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, lrs):
        return self.forward_core(lrs)
    
class iconVSRFusion(iconVSR):
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

