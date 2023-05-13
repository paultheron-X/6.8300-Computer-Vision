import torch
import torch.nn as nn
from torch.nn import init


class CustomAttentionSingleHead(nn.Module):
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
        return elemwise_mult


class CustomAttentionFeature(nn.Module):
    def __init__(self, num_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_channels = num_channels

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

    def forward(self, input):
        # input is supposed to be of shape (batch_size, 3, num_channels, height, width) (elementwise_mult)
        # print('input shape', input.shape)

        # apply one conv3D on the 3D block, so that it stays the same size
        block3d_f = self.relu(self.conv3d_feat(input))
        # print('block3d_f shape', block3d_f.shape)

        # concat the 3D block with the block3d of the beginning, on the channel dimension
        concat = torch.cat([input, block3d_f], dim=1)
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


class DeepCustomAttentionFeature(nn.Module):
    def __init__(self, num_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_channels = num_channels

        # 3D conv layers
        self.conv3d_feat = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )

        self.conv3d_feat_bis = nn.Conv3d(
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

        self.conv2d_feat_bis = nn.Conv2d(
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

    def forward(self, input):
        # input is supposed to be of shape (batch_size, 3, num_channels, height, width) (elementwise_mult)
        # print('input shape', input.shape)

        # apply one conv3D on the 3D block, so that it stays the same size
        block3d_f = self.relu(self.conv3d_feat(input))
        block3d_f = self.relu(self.conv3d_feat_bis(block3d_f))

        # concat the 3D block with the block3d of the beginning, on the channel dimension
        concat = torch.cat([input, block3d_f], dim=1)
        # print('concat shape', concat.shape)

        # apply the last conv3d, to get an output of shape (batch_size, num_channels, 1, height, width)
        block2d = self.relu(self.conv3d_compress(concat))
        # print('block2d shape', block2d.shape)

        # squeeze the block2d to have a shape of (batch_size, num_channels, height, width)
        block2d = block2d.squeeze(2)
        # print('block2d shape', block2d.shape)

        # apply a conv2d on the block2d to get a shape of (batch_size, num_channels, height, width)
        block2d_f = self.relu(self.conv2d_feat(block2d))
        block2d_f = self.relu(self.conv2d_feat_bis(block2d_f))
        # print('block2d_f shape', block2d_f.shape)

        # concat the block2d with the block2d_f on the channel dimension
        concat = torch.cat([block2d, block2d_f], dim=1)
        # print('concat shape', concat.shape)

        # apply the last conv2d, to get an output of shape (batch_size, num_channels, height, width)
        output = self.relu(self.conv2d_compress(concat))
        # print('output shape', output.shape)

        return output


class CustomAttention(nn.Module):
    def __init__(self, num_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_channels = num_channels

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.attn = CustomAttentionSingleHead(num_channels=self.num_channels)

        self.attn_feat = CustomAttentionFeature(num_channels=self.num_channels)

        # init everything
        self._initialize_weights()

    def _initialize_weights(self):
        scale = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        elemwise_mult = self.attn(input)
        block3d = elemwise_mult.permute(0, 2, 1, 3, 4)
        output = self.attn_feat(block3d)
        return output


class MultiHeadCustomAttention(nn.Module):
    def __init__(self, num_channels, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_channels = num_channels
        self.num_heads = num_heads

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 1step: Apply a conv 2D on each of the 3 input frames, to project them onto an other dimension space: projection is shared among the 3 frames on a single head
        self.heads_projector = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.num_channels,
                    out_channels=self.num_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(self.num_heads)
            ]
        )  # Here one head projects the 3 frames onto the same space

        # 2nd step: repeat the attention block: 2D conv + matrix M + pointwise multiplication
        self.heads = nn.ModuleList(
            [
                CustomAttentionSingleHead(num_channels=self.num_channels)
                for _ in range(self.num_heads)
            ]
        )
        # 3rd step: on that result, combine with a 3D conv, to get a 3D block of the same shape as before
        self.heads_combiner = nn.Conv3d(
            in_channels=self.num_heads * self.num_channels,
            out_channels=self.num_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )
        # 4th step: do the final block: 3D conv + 2D conv
        self.final_block = CustomAttentionFeature(num_channels=self.num_channels)

    def forward(self, input):
        # 1st step,
        # apply a conv2d on each of the 3 input frames, to project them onto an other dimension space: projection is shared among the 3 frames on a single head

        in_1 = input[0]
        in_2 = input[1]
        in_3 = input[2]

        print("in_1 shape", in_1.shape)
        print("in_2 shape", in_2.shape)
        print("in_3 shape", in_3.shape)

        # project in_1 in_2, in_3 onto all the heads
        in_1_projected = [self.relu(head(in_1)) for head in self.heads_projector]
        in_2_projected = [self.relu(head(in_2)) for head in self.heads_projector]
        in_3_projected = [self.relu(head(in_3)) for head in self.heads_projector]

        # 2nd step, pass in the real heads
        elem_wise_tens = [
            head((in_1_projected[i], in_2_projected[i], in_3_projected[i]))
            for i, head in enumerate(self.heads)
        ]

        # concat the elem_wise_tens on the channel dimension 2 here
        concatenates = torch.cat(elem_wise_tens, dim=2).permute(0, 2, 1, 3, 4)

        # 3rd step, combine the heads
        output = self.relu(self.heads_combiner(concatenates))

        # 4th step
        output = self.final_block(output)

        return output
