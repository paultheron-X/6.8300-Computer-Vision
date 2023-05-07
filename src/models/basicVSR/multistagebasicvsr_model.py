import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from .basicvsr_model import basicVSR


class MultiStageBasicVSR(basicVSR):
    def __init__(self, num_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attention = MultiheadAttention(embed_dim=self.mid_channels, num_heads=num_heads)
        self.rolling_window = kwargs.get('rolling_window', 5)
        self.mid_frame = self.rolling_window // 2           

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
        out_core_fwd_1, out_core_bwd_1 = self.forward_core(input_1)
        out_core_fwd_2, out_core_bwd_2 = self.forward_core(input_2)
        out_core_fwd_3, out_core_bwd_3 = self.forward_core(input_3)
        
        output_1 = []
        output_2 = []
        output_3 = []
        for i in range(self.rolling_window):
            out_1 = self.forward_fusion(out_core_fwd_1[i], out_core_bwd_1[i])
            output_1.append(out_1)
            out_2 = self.forward_fusion(out_core_fwd_2[i], out_core_bwd_2[i])
            output_2.append(out_2)
            out_3 = self.forward_fusion(out_core_fwd_3[i], out_core_bwd_3[i])
            output_3.append(out_3)
        
        output_1 = torch.stack(output_1, dim=1)
        output_2 = torch.stack(output_2, dim=1)
        output_3 = torch.stack(output_3, dim=1)
        
        print('output_1', output_1.shape)
        print('output_2', output_2.shape)
        print('output_3', output_3.shape)

        # Concatenate the outputs along the time dimension and apply multi-head attention
        output_concat = torch.cat([output_1, output_2, output_3], dim=1)
        print('output_concat', output_concat.shape)
        output_concat = output_concat.permute(1, 0, 2, 3, 4).reshape(output_concat.size(1), output_concat.size(0), -1)
        print('output_concat', output_concat.shape)
        output_attention, _ = self.multihead_attention(output_concat, output_concat, output_concat)
        output_attention = output_attention.view(output_concat.size(0), output_concat.size(1), input.size(2), input.size(3), input.size(4)).permute(1, 0, 2, 3, 4)

        # Upsample the middle frame
        output = self.upsample(output_attention[:, self.mid_frame, :, :,:])
        output = self.lrelu(self.upsample1(output))
        output = self.lrelu(self.upsample2(output))
        output = self.lrelu(self.conv_hr(output))
        output = self.conv_last(output)

        return output
