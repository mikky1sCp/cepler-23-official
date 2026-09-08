def ray_attention_flops_hook(module, input, output):
    x = input[0]
    batch, seq, d = x.shape
    rays = module.num_rays
    flops = 2 * batch * d * rays * seq  # x_perm * ray_weights
    if module.sparse_rays:
        flops += 2 * batch * rays * seq
    flops += 3 * batch * rays * seq      # softmax (dim=2)
    flops += 2 * batch * rays * d * seq  # weighted_sum
    flops += 3 * batch * rays * seq      # softmax (dim=1)
    flops += 2 * batch * seq * rays * d  # out
    flops += 2 * batch * seq * d * d     # W_o
    module.__flops__ = flops

def lightweight_ffn_flops_hook(module, input, output):
    x = input[0]
    batch, seq, d = x.shape
    rank = module.fc1.out_features
    d_ff = module.fc2.out_features
    flops = 2 * batch * seq * d * rank       # fc1
    flops += 2 * batch * seq * rank * d_ff   # fc2
    flops += 2 * batch * seq * d_ff * d      # fc3
    module.__flops__ = flops

def standard_ffn_flops_hook(module, input, output):
    x = input[0]
    batch, seq, d = x.shape
    d_ff = module.fc1.out_features
    flops = 2 * batch * seq * d * d_ff       # fc1
    flops += 2 * batch * seq * d_ff * d      # fc2
    module.__flops__ = flops

def multihead_attention_flops_hook(module, input, output):
    x = input[0]
    batch, seq, d = x.shape
    heads = module.num_heads
    d_k = module.d_k
    flops = 3 * 2 * batch * seq * d * d          # Q,K,V проекции
    flops += 2 * batch * heads * seq * seq * d_k # Q*K^T
    flops += 3 * batch * heads * seq * seq       # softmax
    flops += 2 * batch * heads * seq * seq * d_k # attn*V
    flops += 2 * batch * seq * d * d             # W_o
    module.__flops__ = flops
