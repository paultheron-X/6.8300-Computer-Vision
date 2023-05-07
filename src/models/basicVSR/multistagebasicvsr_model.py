import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from .basicvsr_model import basicVSR


class MultiStageBasicVSR(basicVSR):
    def __init__(self, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attention = MultiheadAttention(embed_dim=kwargs['mid_channels'], num_heads=num_heads)
        self.mid_frame = kwargs['rolling_window'] // 2            

    def forward(self, input):
        '''
        We're only doing the forward duch that we predict on the middle frame
        
        if the frames are I1, I2, I3, I4, I5, I6, I7
        input1 is all the frames
        input2 is all the frames except every other frame (have to be the middle)
        input3 is all the frames except every third frame (have to be the middle)
        '''
        
        # the split of frames is handled by the dataloader
        input_1, input_2, input_3 = input
        
        # all the input are of shape (batch_size, num_frames, num_channels, height, width)

        # Process each input with the same BasicVSRCoreFusion instance
        output_1 = self.forward_fusion(input_1)
        output_2 = self.forward_fusion(input_2)
        output_3 = self.forward_fusion(input_3)

        # Concatenate the outputs along the time dimension and apply multi-head attention
        output_concat = torch.cat([output_1, output_2, output_3], dim=1)
        output_concat = output_concat.permute(1, 0, 2, 3, 4).reshape(output_concat.size(1), output_concat.size(0), -1)
        output_attention, _ = self.multihead_attention(output_concat, output_concat, output_concat)
        output_attention = output_attention.view(output_concat.size(0), output_concat.size(1), input.size(2), input.size(3), input.size(4)).permute(1, 0, 2, 3, 4)

        # Upsample the middle frame
        output = self.upsample(output_attention[:, self.mid_frame, :, :,    :])
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        return upsampled_output
