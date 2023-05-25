import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from models.networks.architecture import ResidualBlock
from models.networks.architecture import SPADE
from models.networks.architecture import PositionalNorm2d
import util.util as util
from models.networks.modules import SelfAttentionLayer, FFNLayer, AdaBlock

class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input

def dynamic_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    q = q.view(b, c_qk, h_q * w_q).transpose(-1, -2).contiguous()
    k = k.view(b, c_qk, h_kv * w_kv)
    v = v.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
    q_prune = q_prune.view(b, -1, h_q * w_q).transpose(-1, -2).contiguous()
    k_prune = k_prune.view(b, -1, h_kv * w_kv)
    mask = SignWithSigmoidGrad.apply(torch.bmm(q_prune, k_prune) / k_prune.shape[1])
    # q: b, N_q, c_qk
    # k: b, c_qk, N_kv
    # v: b, N_kv, c_v
    if smooth is None:
        smooth = c_qk ** 0.5
    cor_map = torch.bmm(q, k) / smooth
    attn = torch.softmax(cor_map, dim=-1)
    # attn: b, N_q, N_kv
    masked_attn = attn * mask
    output = torch.bmm(masked_attn, v)
    # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = masked_attn.sum(-1).view(b, 1, h_q, w_q)

    # conf_map = torch.max(cor_map, -1, keepdim=True)[0]
    # conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(b, 1, h_q, w_q)
    # conf_map = torch.sigmoid(conf_map * 10.0)

    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        output2 = torch.bmm(torch.softmax(torch.masked_fill(cor_map, mask.bool(), -1e4), dim=-1),
                            v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    else:
        output2 = None
    return output, cor_map, conf, output2

def Multi_head_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None, num_heads=8):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    c_v = v.shape[1]
    q = q.reshape(b, num_heads, c_qk // num_heads, h_q * w_q).permute(0, 1, 3, 2).contiguous()
    k = k.reshape(b, num_heads, c_qk // num_heads, h_kv * w_kv)
    v = v.reshape(b, num_heads, c_v // num_heads, h_kv * w_kv).permute(0, 1, 3, 2).contiguous()

    q_prune = q_prune.view(b, -1, h_q * w_q).transpose(-1, -2).contiguous()
    k_prune = k_prune.view(b, -1, h_kv * w_kv)
    mask = SignWithSigmoidGrad.apply(torch.bmm(q_prune, k_prune) / k_prune.shape[1])
    # q: b, N_q, c_qk
    # k: b, c_qk, N_kv
    # v: b, N_kv, c_v
    if smooth is None:
        smooth = c_qk ** 0.5
    cor_map = (q @ k) / smooth
    attn = torch.softmax(cor_map, dim=-1)
    # attn: b, N_q, N_kv
    masked_attn = attn * mask
    output = (masked_attn @ v)
    # # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = torch.mean(masked_attn, dim=1).sum(-1).view(b, 1, h_q, w_q)

    # conf_map = torch.max(cor_map, -1, keepdim=True)[0]
    # conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(b, 1, h_q, w_q)
    # conf_map = torch.sigmoid(conf_map * 10.0)

    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        output2 = torch.bmm(torch.softmax(torch.masked_fill(torch.mean(cor_map, dim=1), mask.bool(), -1e4), dim=-1),
                            v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    else:
        output2 = None
    return output, cor_map, conf, output2


class DynamicTransformerBlock(nn.Module):
    def __init__(self, embed_dim_qk, embed_dim_v, dim_prune, ic, smooth=None):
        super().__init__()
        self.f = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.g = nn.Conv2d(embed_dim_qk, embed_dim_qk, (1, 1), (1, 1))
        self.h = nn.Conv2d(embed_dim_v, embed_dim_v, (1, 1), (1, 1))
        self.f_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.g_prune = nn.Conv2d(embed_dim_qk, dim_prune, (1, 1), (1, 1))
        self.spade = SPADE(embed_dim_v, ic)
        # self.res_block = ResidualBlock(embed_dim_v)
        self.smooth = smooth
        self.adablock = AdaBlock(embed_dim_v)
    # x4, kv4, kv4, pos, seg_map, F.avg_pool2d(
    def forward(self, q, k, v, pos, seg_map, v2=None):

        pos = pos.repeat(q.shape[0], 1, 1, 1)
        query = torch.cat([util.feature_normalize(q), pos], dim=1)
        key = torch.cat([util.feature_normalize(k), pos], dim=1)
        attn_output, cor_map, conf, output2 = dynamic_attention(
            self.f(query), self.g(key), self.f_prune(query), self.g_prune(key), self.h(v), self.smooth, v2)
        spade_output = self.spade(q, seg_map)
        # out = warp + (1-conf) * seg
        # warp +
        y = PositionalNorm2d(attn_output + (1 - conf) * spade_output + q)
        y = self.adablock(y)

        return y, output2, cor_map


