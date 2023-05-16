import torch
import torch.nn as nn
from .basicvsr_model import basicVSR
from .attention_modules import CustomAttention, MultiHeadCustomAttention

from torch.nn import init
import logging


class MultiStageBasicMhead(basicVSR):
    def __init__(self, pretrained_bvsr=None, pretrained_model=None, *args, **kwargs):
        super().__init__(pretrained_model=pretrained_bvsr, *args, **kwargs)

        # freeze the weights of the basicVSR if those params are before the fusion layer
        continue_freeze = True
        for k, v in self.named_parameters():
            if continue_freeze:
                v.requires_grad_(False)
            if "fusion.bias" in k:
                continue_freeze = False
        self.attention = MultiHeadCustomAttention(
            num_heads=kwargs.get("num_heads", 4), num_channels=self.mid_channels
        )
        self.attention_output_bn = nn.BatchNorm2d(self.mid_channels)
        self.upsample1_bn = nn.BatchNorm2d(self.mid_channels)
        self.upsample2_bn = nn.BatchNorm2d(64)
        self.conv_hr_bn = nn.BatchNorm2d(64)
        self.rolling_window = kwargs.get("rolling_window", 5)
        self.mid_frame = self.rolling_window // 2

        # reinit the weights of the upsample1, upsample2, conv_hr, conv_last
        # self.upsample1.init_weights()
        # self.upsample2.init_weights()
        # self.conv_hr.apply(self._initialize_weights)
        # self.conv_last.apply(self._initialize_weights)

        if pretrained_model is not None:
            if pretrained_model.endswith(".pth"):
                logging.info(f"Loading pretrained model from {pretrained_model}")
                self.load_pretrained_weights_mstage(torch.load(pretrained_model))

    def load_pretrained_weights_mstage(self, weights_pret):
        # assert that there is the same number of keys
        assert len(weights_pret.keys()) == len(self.state_dict().keys())
        model_p = self.state_dict()
        list_pret = list(weights_pret.keys())
        # load everything with a loop, don't care about the keys
        for i, k in enumerate(model_p.keys()):
            model_p[k] = weights_pret[list_pret[i]]
        self.load_state_dict(model_p)

    def _initialize_weights(self, m):
        scale = 0.1
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, input):
        """
        We're only doing the forward duch that we predict on the middle frame

        if the frames are I1, I2, I3, I4, I5, I6, I7
        input1 is all the frames
        input2 is all the frames except every other frame (have to be the middle)
        input3 is all the frames except every third frame (have to be the middle)
        """

        # the split of frames is handled by the dataloader
        input_1, input_2, input_3 = input

        # all the input are of shape (batch_size, num_frames, num_channels, height, width)

        # Process each input with the same BasicVSRCoreFusion instance
        out_core_fwd_1, out_core_bwd_1 = self.forward_core(input_1)
        out_core_fwd_2, out_core_bwd_2 = self.forward_core(input_2)
        out_core_fwd_3, out_core_bwd_3 = self.forward_core(input_3)

        # output_1 = []
        # output_2 = []
        # output_3 = []
        # for i in range(self.rolling_window):
        #    out_1 = self.forward_fusion(out_core_fwd_1[i], out_core_bwd_1[i])
        #    output_1.append(out_1)
        #    out_2 = self.forward_fusion(out_core_fwd_2[i], out_core_bwd_2[i])
        #    output_2.append(out_2)
        #    out_3 = self.forward_fusion(out_core_fwd_3[i], out_core_bwd_3[i])
        #    output_3.append(out_3)

        # keep only the middle frame, and concatenate to have a tensor of shape (batch_size, 3, num_channels, height, width)
        output_1 = self.forward_fusion(
            out_core_fwd_1[self.mid_frame], out_core_bwd_1[self.mid_frame]
        )
        output_2 = self.forward_fusion(
            out_core_fwd_2[self.mid_frame], out_core_bwd_2[self.mid_frame]
        )
        output_3 = self.forward_fusion(
            out_core_fwd_3[self.mid_frame], out_core_bwd_3[self.mid_frame]
        )

        # print('output_1', output_1.shape)
        # print('output_2', output_2.shape)
        # print('output_3', output_3.shape)

        # print('output', output.shape)

        attention_output = self.attention((output_1, output_2, output_3))
        attention_output_bn = self.attention_output_bn(attention_output)
        out = self.lrelu(self.upsample1_bn(self.upsample1(attention_output_bn)))
        out = self.lrelu(self.upsample2_bn(self.upsample2(out)))
        out = self.lrelu(self.conv_hr_bn(self.conv_hr(out)))
        out = self.conv_last(out)
        out_temp = out
        base = self.img_upsample(input_1[:, self.mid_frame, :, :, :])
        out += base

        return out
