import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_self_attention(Q, K, V, n_heads=1, causal=True):
    """
    Self-attention block.

    Note: You will keep coming back to this function and fill in more of it
    after completing steps 1-3! Make sure that once you're done, all the tests should pass.

    TODO: fill in the missing blocks of this function

    :return: A tensor containing the result of the self-attention operation.
    """
    assert Q.shape == K.shape == V.shape
    B, n_tok, n_embd = Q.size()

    # TODO: Step 3 -- split heads.
    if n_heads > 1:
        Q, K, V = split_heads_qkv(Q, K, V, n_heads)

    # TODO: Step 1 -- calculate raw attetion.
    # Hint: you need two lines here.
    A = pairwise_similarities(Q, K)
    A = attn_scaled(A, n_embd, n_heads)

    # TODO: Step 2 -- create and apply the causal mask to attention.
    if causal:
        mask = make_causal_mask(n_tok)
        A = apply_causal_mask(mask, A)

    # TODO: Step 1 -- softmax the raw attention and use it to get outputs.
    # Hint: you need two lines here.
    if causal:
        Z = attn_softmax(A * mask + ((1 - mask) * -torch.inf).nan_to_num(nan=0.0))
        A = attn_softmax(A, mask)
    else:
        A = attn_softmax(A)
    y = compute_outputs(A, V)

    # TODO: Step 3 -- merge heads.
    if n_heads > 1:
        y = merge_heads(y)


    # output should have the same shape as input
    assert y.shape == (B, n_tok, n_embd), f"output shape should be {y.shape}, but is actually {(B, n_tok, n_embd)}"
    return y

# Step 1: Implement the core components of attention.


def pairwise_similarities(Q, K):
    """
    Dot product attention is computed via the dot product between each query and each key.
    :return: The raw attention scores, A = QK^T.
    """
    return Q @ K.transpose(-2, -1)

def attn_scaled(A, n_embd:float, n_heads:float):
    """
    Scale the raw attention scores.
    :return: Scaled raw attention scores.
    """
    assert n_embd % n_heads == 0, "d must be divisible by number of heads"
    return A / torch.sqrt(torch.tensor(n_embd/n_heads))

def attn_softmax(A, mask=None):
    """
    Normalize the scaled raw attention scores with softmax.
    :return: Normalized attention scores, A' = softmax(A).

    You are allowed (and encouraged) to use a pytorch softmax implementation here.
    """
    # Hint: the softmax function should be applied to dim=-1.
    if mask is not None:
        return torch.softmax(A, -1) * mask
    else:
        return torch.softmax(A, -1)

def compute_outputs(A, V):
    """
    Get outputs as a weighted sum of values by attention scores, using matrices.
    :return: Output, O = AV.
    """
    return A @ V

# Step 2: Implement causal masking for language modeling.

def make_causal_mask(n_tok:int):
    """
    Create a mask matrix that masks future context for the attention.
    :return: A mask matrix which is a tensor of shape (n_tok, n_tok)
    """
    # Hint: In order for your experiments to work properly later, you'll need to put `.to(DEVICE)` at
    # the end of your expression for this. This will not be relevant until section 2.2.
    return torch.tril(torch.ones((n_tok, n_tok))).to(DEVICE)

def apply_causal_mask(mask, A):
    """
    Apply mask to attention.
    :return: A masked attention matrix.
    """
    return A * mask

# Step 3: Implement multi-head attention.

def split_heads_qkv(Q, K, V, n_heads:int):
    """
    Provided as a utility -- you can choose to not use it if you'd like.
    """
    return (split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads))

def split_heads(x, n_heads:int):
    """
    Splitting x across multiple heads.
    :return: A splitted x.
    """
    B, n_tok, n_embd = x.size()
    assert n_embd % n_heads == 0, "d must be divisible by number of heads"
    x = x.reshape(B, n_tok, n_heads, n_embd // n_heads)
    return x.transpose(1, 2)

def merge_heads(y):
    """
    Reversing splitting action of y.
    :return: A merged y.
    """
    B, nh, n_tok, nc = y.size()
    y = y.transpose(1, 2)
    return y.reshape(B, n_tok, nh * nc)