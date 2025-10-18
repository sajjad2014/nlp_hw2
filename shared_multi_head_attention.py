import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_shared_qkv(n_embd:int):
    """
    This function is given to you.
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    qkv = nn.Linear(n_embd, n_embd)
    return (qkv, qkv, qkv)

def init_shared_qk(n_embd:int):
    """
    This function is given to you.
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    qk = nn.Linear(n_embd, n_embd)
    v = nn.Linear(n_embd, n_embd)
    return (qk, qk, v)

def init_shared_qv(n_embd:int):
    """
    This function is given to you.
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    qv = nn.Linear(n_embd, n_embd)
    k = nn.Linear(n_embd, n_embd)
    return (qv, k, qv)


def init_shared_kv(n_embd:int):
    """
    This function is given to you.
    :return: A tuple of length 3 containing the projections for Q, K, V.
    """
    q = nn.Linear(n_embd, n_embd)
    kv = nn.Linear(n_embd, n_embd)
    return (q, kv, kv)