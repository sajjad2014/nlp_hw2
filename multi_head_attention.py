import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wrapper for Torch attention implementation - do not modify
def MHA_wrapper(query, key, value, n_heads=1, causal=False):
    """
    This is a wrapper around the PyTorch implementation of multi-head attention.
    You will use this implementation to compare to your implementation for code testing.
    """
    assert query.shape == key.shape == value.shape
    _, n_tok, n_embd = query.shape

    query = query.transpose(0,1)
    key = key.transpose(0,1)
    value = value.transpose(0,1)

    in_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device).repeat((3, 1))
    out_proj_weight = torch.eye(n_embd, dtype=key.dtype, device=key.device)

    attn_mask = None
    if causal:
        attn_mask = torch.tril(torch.ones(n_tok, n_tok, dtype=bool, device=key.device)).logical_not()

    out, _ = F.multi_head_attention_forward(
        query, key, value, n_embd, n_heads,
        in_proj_weight=in_proj_weight, in_proj_bias=None,
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
        out_proj_weight=out_proj_weight, out_proj_bias=None,
        attn_mask=attn_mask, need_weights=False,)

    return out.transpose(0,1)

# testing setup - do not modify
def set_up_tests():

    # make these bigger if you want a stricter test of your code
    part1_n_tok = 10
    part1_n_emb = 6

    # generate fixed pseudo-random Q,K,V for testing attn function
    torch.manual_seed(447)

    # Initialize random testing Q,K,V
    part1_key = torch.randn(1, part1_n_tok, part1_n_emb)
    part1_value = torch.randn(1, part1_n_tok, part1_n_emb)
    part1_query = torch.randn(1, part1_n_tok, part1_n_emb)

    return part1_key, part1_query, part1_value

def init_qkv_proj(n_embd:int):
    """
    This function is given to you.
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    return (nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd), nn.Linear(n_embd, n_embd))


def self_attention(Q, K, V, n_heads=1, causal=True):
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

def attn_softmax(A):
    """
    Normalize the scaled raw attention scores with softmax.
    :return: Normalized attention scores, A' = softmax(A).

    You are allowed (and encouraged) to use a pytorch softmax implementation here.
    """
    # Hint: the softmax function should be applied to dim=-1.
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
    return A * mask + ((1 - mask) * -torch.inf).nan_to_num(nan=0.0)

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

# testing functions for your convenience
# autograder won't call these directly, but this is basically what it will do
# difference tolerance will be 10e-5 for all tests
def test_step_1():
    part1_key, part1_query, part1_value = set_up_tests()
    out_A = self_attention(part1_query, part1_key, part1_value, n_heads=1, causal=False)
    out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=1, causal=False)
    assert out_A.shape == out_B.shape == part1_query.shape, f"Step 1: output shape should be {out_B.shape} but is actually {out_A.shape}"
    max_abs_diff = (out_A - out_B).abs().max().item()
    print('max diff:', max_abs_diff)
    if max_abs_diff <= 10e-5:
        print("step 1 (scaled attention) correct")
    else:
        print("step 1 (scaled attention) incorrect")

def test_step_2():
    part1_key, part1_query, part1_value = set_up_tests()
    out_A = self_attention(part1_query, part1_key, part1_value, n_heads=1, causal=True)
    out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=1, causal=True)
    assert out_A.shape == out_B.shape == part1_query.shape, f"Step 2: output shape should be {out_B.shape} but is actually {out_A.shape}"
    max_abs_diff = (out_A - out_B).abs().max().item()
    print('max diff:', max_abs_diff)
    if max_abs_diff <= 10e-5:
        print("step 2 (causal masking) correct")
    else:
        print("step 1 (causal masking) incorrect")

def test_step_3():
    part1_key, part1_query, part1_value = set_up_tests()
    out_A = self_attention(part1_query, part1_key, part1_value, n_heads=3, causal=True)
    out_B = MHA_wrapper(part1_query, part1_key, part1_value, n_heads=3, causal=True)
    assert out_A.shape == out_B.shape == part1_query.shape, f"Step 3: output shape should be {out_B.shape} but is actually {out_A.shape}"
    max_abs_diff = (out_A - out_B).abs().max().item()
    print('max diff:', max_abs_diff)
    if max_abs_diff <= 10e-5:
        print("step 1 (multi-head attention) correct")
    else:
        print("step 1 (multi-head attention) incorrect")

def test_all():
    test_step_1()
    test_step_2()
    test_step_3()


# run this file as a script to test your whole implementation
if __name__ == "__main__":
    DEVICE = 'cpu'
    test_all()