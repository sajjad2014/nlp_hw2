import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def simple_attention(Q, K, V, n_heads=1, causal=True):
    """
    implementation of Simple Attention Free Transformer
    """
    assert Q.shape == K.shape == V.shape
    B, n_tok, n_embd = Q.size()

    # if causal:
    #     mask = torch.tril(torch.ones((n_tok, n_tok))).to(DEVICE)
    #     print(K.shape)
    #     K = K * mask + ((1 - mask) * -torch.inf).nan_to_num(nan=0.0)
    weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
    Q_sig = torch.sigmoid(Q)
    Yt = torch.mul(Q_sig, weights)

    if causal:
        K_exp = torch.exp(K)
        V_sum = torch.cumsum(K_exp * V, dim=1).diagonal(dim1=1, dim2=2)
        V_sum = V_sum.unsqueeze(1)
        norm_weight = torch.cumsum(K_exp, dim=1).diagonal(dim1=1, dim2=2)
        norm_weight = norm_weight.unsqueeze(1)
        V_sum = V_sum / norm_weight
    else:
        V_sum = torch.sum(torch.softmax(K, 1) * V, dim=1, keepdim=True)
    y = torch.sigmoid(Q) * V_sum

    # output should have the same shape as input
    assert y.shape == (B, n_tok, n_embd), f"output shape should be {y.shape}, but is actually {(B, n_tok, n_embd)}"
    return y
