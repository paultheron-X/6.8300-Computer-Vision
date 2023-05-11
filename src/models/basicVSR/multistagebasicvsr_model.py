import torch
import torch.nn as nn
from .basicvsr_model import basicVSR

from torch.nn import init



class CustomAttention(nn.Module):
    def __init__(self, num_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_channels = num_channels

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # first conv layer: we apply it on each elem of dim 1 of the input of shape (batch_size, 3, num_channels, height, width), so we have to apply it on each frame
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 3D conv layers
        self.conv3d_feat = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )
        self.conv3d_compress = nn.Conv3d(
            in_channels=2 * self.num_channels,
            out_channels=self.num_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        # 2D conv layers
        self.conv2d_feat = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2d_compress = nn.Conv2d(
            in_channels=2 * self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        # init everything
        self._initialize_weights()
    
    def _initialize_weights(self):
        scale = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        # input is of shape (batch_size, 3, num_channels, height, width), apply the conv on each frame

        first_conv = self.relu(self.conv1(input[0])).unsqueeze(1)
        second_conv = self.relu(self.conv2(input[1])).unsqueeze(1)
        third_conv = self.relu(self.conv3(input[2])).unsqueeze(1)
        # print(first_conv.shape)
        # print(second_conv.shape)
        # print(third_conv.shape)

        # concat the results
        concat_input = torch.cat([first_conv, second_conv, third_conv], dim=1)
        # print('concat shape', concat.shape)

        # calculate the M1, M2, M3 attention masks for each frame
        exp_concat = torch.exp(concat_input)

        # calculate the attention masks
        M = exp_concat / torch.sum(exp_concat, dim=1).unsqueeze(1)
        # print('M shape', M.shape)

        input_cat = torch.cat(
            [input[0].unsqueeze(1), input[1].unsqueeze(1), input[2].unsqueeze(1)], dim=1
        )

        # calculate the pointwise multiplication of the attention masks with the input
        elemwise_mult = torch.mul(
            M, input_cat
        )  # shape (batch_size, 3, num_channels, height, width)
        # print('elemwise_mult shape', elemwise_mult.shape)

        # flip the num_channels and 3 dimensions
        block3d = elemwise_mult.permute(0, 2, 1, 3, 4)
        # print('block3d shape', block3d.shape)

        # apply one conv3D on the 3D block, so that it stays the same size
        block3d_f = self.relu(self.conv3d_feat(block3d))
        # print('block3d_f shape', block3d_f.shape)

        # concat the 3D block with the block3d of the beginning, on the channel dimension
        concat = torch.cat([block3d, block3d_f], dim=1)
        # print('concat shape', concat.shape)

        # apply the last conv3d, to get an output of shape (batch_size, num_channels, 1, height, width)
        block2d = self.relu(self.conv3d_compress(concat))
        # print('block2d shape', block2d.shape)

        # squeeze the block2d to have a shape of (batch_size, num_channels, height, width)
        block2d = block2d.squeeze(2)
        # print('block2d shape', block2d.shape)

        # apply a conv2d on the block2d to get a shape of (batch_size, num_channels, height, width)
        block2d_f = self.relu(self.conv2d_feat(block2d))
        # print('block2d_f shape', block2d_f.shape)

        # concat the block2d with the block2d_f on the channel dimension
        concat = torch.cat([block2d, block2d_f], dim=1)
        # print('concat shape', concat.shape)

        # apply the last conv2d, to get an output of shape (batch_size, num_channels, height, width)
        output = self.relu(self.conv2d_compress(concat))
        # print('output shape', output.shape)

        return output


class MultiStageBasicVSR(basicVSR):
    def __init__(self, num_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # freeze the weights of the basicVSR if those params are before the fusion layer
        continue_freeze = True
        for k, v in self.named_parameters():
            if continue_freeze:
                v.requires_grad_(False)
            if "fusion.bias" in k:
                continue_freeze = False
        self.attention = CustomAttention(num_channels=self.mid_channels)
        self.rolling_window = kwargs.get("rolling_window", 5)
        self.mid_frame = self.rolling_window // 2
        
        # reinit the weights of the upsample1, upsample2, conv_hr, conv_last
        self.upsample1.init_weights()
        self.upsample2.init_weights()
        self.conv_hr.apply(self._initialize_weights)
        self.conv_last.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        scale = 0.1
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
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

        out = self.lrelu(self.upsample1(attention_output))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        out_temp = out
        base = self.img_upsample(input_1[:, self.mid_frame, :, :, :])
        out += base

        return out #, attention_output #, base, out_temp, (output_1, output_2, output_3)

