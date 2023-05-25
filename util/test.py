import torch

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

def dynamic_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None, num_heads=8):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    q = q.view(b, c_qk, h_q * w_q).transpose(-1, -2).contiguous()

    k = k.view(b, c_qk, h_kv * w_kv)
    v = v.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
    q_prune = q_prune.view(b, -1, h_q * w_q).transpose(-1, -2).contiguous()
    k_prune = k_prune.view(b, -1, h_kv * w_kv)
    mask = SignWithSigmoidGrad.apply(torch.bmm(q_prune, k_prune) / k_prune.shape[1])
    print(mask)
    print(mask.bool())
    # q: b, N_q, c_qk
    # k: b, c_qk, N_kv
    # v: b, N_kv, c_v
    if smooth is None:
        smooth = c_qk ** 0.5
    cor_map = torch.bmm(q, k) / smooth
    out = torch.masked_fill(cor_map, mask.bool(), -1e4)
    print(out.shape)
    print(out)
    assert 0 > 1
    attn = torch.softmax(cor_map, dim=-1)
    attn[attn < torch.topk(attn, k=4, dim=-1)[0][..., -1, None]] = 0.
    print(attn)
    conf_map = 1.0 - attn
    print(conf_map)

    # attn: b, N_q, N_kv
    masked_attn = attn * mask
    output = torch.bmm(masked_attn, v)
    # # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = masked_attn.sum(-1).view(b, 1, h_q, w_q)
    print(conf.shape)
    print(conf)
    # conf_map = torch.max(cor_map, -1, keepdim=True)[0]
    # conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(b, 1, h_q, w_q)
    # conf_map = torch.sigmoid(conf_map * 10.0)

    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        output2 = torch.bmm(attn, v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    else:
        output2 = None
    return output, cor_map, conf, output2

def dynamic_mutial_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None, num_heads=8):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    c_v = v.shape[1]
    q = q.reshape(b, num_heads, c_qk//num_heads, h_q*w_q).permute(0, 1, 3, 2).contiguous()
    k = k.reshape(b, num_heads, c_qk//num_heads, h_kv*w_kv)
    v = v.reshape(b, num_heads, c_v//num_heads, h_kv*w_kv).permute(0, 1, 3, 2).contiguous()
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
    print(masked_attn.shape)
    output = (masked_attn @ v)
    # # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = torch.mean(masked_attn, dim=1).sum(-1).view(b, 1, h_q, w_q)
    print(conf.shape)
    # conf_map = torch.max(cor_map, -1, keepdim=True)[0]
    # conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(b, 1, h_q, w_q)
    # conf_map = torch.sigmoid(conf_map * 10.0)

    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        mat = torch.softmax(torch.masked_fill(torch.mean(cor_map, dim=1), mask.bool(), -1e4), dim=-1)
        print(mat.shape)
        output2 = torch.bmm(mat, v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)

    else:
        output2 = None
    print('output2', output2.shape)
    return output, cor_map, conf, output2

if __name__ == '__main__':
    q = torch.randn((1, 128, 32, 32))
    # b, (H_q // 2) * (W_q // 2), 4, c
    # x = torch.rand((1, 64*64, 4, 128))
    # x = x.view((1, 4, 128, 4096))
    # mask = torch.rand((1, 4096, 4096))
    # x = x*mask
    # print(x.shape)
    import torch.nn.functional as F
    k = q
    v = torch.rand((1, 256, 32, 32))
    v2 = torch.rand((1, 3, 32, 32))
    output, cor_map, conf_map, output2 = dynamic_mutial_attention(q, k, q, k, v, v2=v2)
    x = torch.rand((1024, 4, 64))
    y = torch.ones((1, 1024, 1024))
    # _, y = torch.topk(y, k=4, dim=-1)
    # z = x[y.view(-1), :, :]
    # print(z.shape) # 1 512 1024  4*128  32*32
