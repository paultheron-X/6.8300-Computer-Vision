import torch
from torch import nn
from torch.nn import MultiheadAttention

from .basicvsr_model import basicVSR

from ..optical_flow.SPyNet import get_spynet
from .modules import flow_warp

import logging

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiHeadAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads, dropout)
    
    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        q = k = v = x
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.permute(1, 0, 2, 3, 4)
        return attn_output

class BasicVSRWithAttention(basicVSR):
    def __init__(self, *args, num_attention_heads=4, **kwargs):
        super().__init__(*args, **kwargs)

        self.backward_attention = CustomMultiHeadAttention(self.mid_channels, num_attention_heads)
        self.forward_attention = CustomMultiHeadAttention(self.mid_channels, num_attention_heads)
        
    def load_pretrained_weights(self, weights_pret):
        model_keys = [k for k in self.state_dict().keys()]
        pretrained_keys = [k for k in weights_pret.keys()]

        new_dict = {}
        if self.optical_module_name == "SPyNet":
            for i, a in enumerate(model_keys):
                if "attention" not in a:
                    new_dict[a] = weights_pret[a]
                else:
                    new_dict[a] = self.state_dict()[a]  # keep the randomly initialized attention weights
        else:
            for i, a in enumerate(model_keys):
                if "optical_module" in a:
                    new_dict[a] = self.state_dict()[a]  # keep the raft weights
                elif "attention" not in a:
                    new_dict[a] = weights_pret[a]
                else:
                    new_dict[a] = self.state_dict()[a]  # keep the randomly initialized attention weights

        self.load_state_dict(new_dict, strict=True)

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, "The height and width of inputs should be at least 64, " f"but got {h} and {w}."

        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            print(feat_prop.shape)
            feat_prop = self.backward_attention(feat_prop)  # apply attention

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            feat_prop = self.forward_attention(feat_prop)  # apply attention

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
