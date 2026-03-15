# =============================================================================
# logits.py — sparse logit buffer + memory-efficient logprob/entropy
#
# Core insight: tokens with prob < SPARSE_LOGIT_THRESH will never be sampled
# in NUM_GENERATIONS completions and contribute epsilon to KL and policy
# gradient. Store only tokens above threshold + one logsumexp scalar per
# position. Memory: ~32MB per sequence vs ~8GB for full logit tensor.
#
# The logsumexp is computed over the FULL distribution before truncation,
# so log probabilities for stored tokens are exact, not approximate.
# =============================================================================

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

from config import ENTROPY_CHUNK, LOGPROB_CHUNK, SPARSE_LOGIT_THRESH


# =============================================================================
# Sparse logit buffer
# =============================================================================

@dataclass
class SparseLogitBuffer:
    """
    Stores logits for a single sequence in sparse format.

    Per token position:
        indices:   (T, K_t)  — vocab indices above threshold (K_t varies per pos)
        values:    (T, K_t)  — logit values for those indices
        logsumexp: (T,)      — logsumexp of FULL distribution (exact partition fn)

    K_t is typically 200-800 for a well-trained model on natural text.
    Stored as a flat padded tensor for GPU efficiency.
    """
    indices:   torch.Tensor   # (T, K) padded with -1
    values:    torch.Tensor   # (T, K) padded with -inf
    logsumexp: torch.Tensor   # (T,)
    seq_len:   int
    K:         int            # padded width


def build_sparse_buffer(
    logits:    torch.Tensor,           # (T, V)  — completion token logits only
    threshold: float = SPARSE_LOGIT_THRESH,
    device:    Optional[torch.device] = None,
    token_chunk: int = 64,             # process this many tokens at a time
) -> SparseLogitBuffer:
    """
    Build sparse buffer from dense logits WITHOUT materializing full (T, V)
    float32 tensors. Processes token_chunk rows at a time.

    Peak memory per chunk: token_chunk × V × 4 bytes
    At token_chunk=64, V=151936: 64 × 151936 × 4 = 38MB — always safe.
    """
    T, V = logits.shape
    if device is None:
        device = logits.device

    # ── Pass 1: compute logsumexp and build sparse indices, chunk by chunk ──
    all_lse     = torch.empty(T, dtype=torch.float32, device=device)
    all_indices = []   # list of 1D tensors (flat token-major sparse indices)
    all_values  = []
    row_nnz     = torch.empty(T, dtype=torch.long, device=device)

    for t_start in range(0, T, token_chunk):
        t_end  = min(t_start + token_chunk, T)
        chunk  = logits[t_start:t_end].float()          # (tc, V)  float32

        # logsumexp for this chunk
        lse_c  = torch.logsumexp(chunk, dim=-1)          # (tc,)
        all_lse[t_start:t_end] = lse_c

        # softmax probabilities for threshold masking
        probs  = torch.exp(chunk - lse_c.unsqueeze(-1))  # (tc, V)
        mask   = probs > threshold                        # (tc, V) bool

        # Always keep top-1 per row
        top1   = probs.argmax(dim=-1, keepdim=True)
        mask.scatter_(1, top1, True)

        # Store sparse indices + values for each row in this chunk
        for i in range(t_end - t_start):
            idx = mask[i].nonzero(as_tuple=False).squeeze(-1)   # (K_i,)
            row_nnz[t_start + i] = idx.shape[0]
            all_indices.append(idx)
            all_values.append(logits[t_start + i, idx])         # bf16 original

        del chunk, probs, mask

    # ── Pass 2: pad into (T, K) tensors ────────────────────────────────────
    K = int(row_nnz.max().item())
    K = max(K, 1)

    indices = torch.full((T, K), -1,    dtype=torch.long,   device=device)
    values  = torch.full((T, K), -1e9,  dtype=logits.dtype, device=device)

    for t, (idx, val) in enumerate(zip(all_indices, all_values)):
        k_t = idx.shape[0]
        indices[t, :k_t] = idx
        values[t,  :k_t] = val

    return SparseLogitBuffer(
        indices=indices,
        values=values,
        logsumexp=all_lse.to(logits.dtype),
        seq_len=T,
        K=K,
    )


def sparse_logprob(
    buf:      SparseLogitBuffer,
    token_id: torch.Tensor,     # (T,) — sampled token ids
) -> torch.Tensor:
    """
    Compute log P(token_id[t] | context) for each position t.
    Exact: log P = logit[t, token_id[t]] - logsumexp[t]

    For tokens outside the sparse set (prob < threshold):
    they should never appear in any sampled completion, so this
    path should never be hit in practice. We return a large
    negative value (-1e9) as a safe fallback.
    """
    T = buf.seq_len
    device = buf.indices.device

    # For each position, find the logit of the sampled token
    # by searching the sparse indices
    token_id = token_id.to(device)
    logit_of_token = torch.full((T,), -1e9, dtype=buf.values.dtype, device=device)

    # Vectorized lookup: compare each position's indices against token_id
    match = (buf.indices == token_id.unsqueeze(-1))  # (T, K)
    found = match.any(dim=-1)                         # (T,)
    match_idx = match.float().argmax(dim=-1)          # (T,) — first match col

    logit_of_token[found] = buf.values[
        torch.arange(T, device=device)[found],
        match_idx[found],
    ]

    return (logit_of_token - buf.logsumexp).to(buf.values.dtype)


def buffer_to_cpu(buf: SparseLogitBuffer) -> SparseLogitBuffer:
    return SparseLogitBuffer(
        indices=buf.indices.cpu(),
        values=buf.values.cpu(),
        logsumexp=buf.logsumexp.cpu(),
        seq_len=buf.seq_len,
        K=buf.K,
    )


def buffer_to_gpu(buf: SparseLogitBuffer, device: torch.device) -> SparseLogitBuffer:
    return SparseLogitBuffer(
        indices=buf.indices.to(device),
        values=buf.values.to(device),
        logsumexp=buf.logsumexp.to(device),
        seq_len=buf.seq_len,
        K=buf.K,
    )


# =============================================================================
# Memory-efficient log softmax — ported from old implementation
# =============================================================================

def _rowwise_logsumexp_chunked(logits_2d: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Numerically stable logsumexp over vocab for logits (T, V) in vocab chunks.
    Avoids allocating a full fp32 (T, V) buffer.
    """
    T = logits_2d.size(0)
    device = logits_2d.device
    m = torch.full((T,), float("-inf"), device=device, dtype=torch.float32)

    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        m  = torch.maximum(m, zc.max(dim=-1).values)

    sum_exp = torch.zeros_like(m)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc      = logits_2d[:, s:s + chunk_size].float()
        sum_exp = sum_exp + torch.exp(zc - m.unsqueeze(-1)).sum(dim=-1)

    return m + torch.log(sum_exp)


def selective_log_softmax(
    logits: torch.Tensor,   # (B, T, V)
    index:  torch.Tensor,   # (B, T)  — token ids
) -> torch.Tensor:
    """
    Compute log P(index[b,t] | context) without storing full (B, T, V) log-prob tensor.
    Used by the training model forward pass (when recompute is needed).

    For our pipeline this is only called when we DON'T have a sparse buffer
    (e.g. reference model logprobs). For policy logprobs we use sparse_logprob().
    """
    if logits.dtype in (torch.float32, torch.float64):
        lse      = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
        selected = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
        return selected - lse
    else:
        token_logprobs = []
        for logits_row, index_row in zip(logits, index):
            lse_row  = _rowwise_logsumexp_chunked(logits_row, LOGPROB_CHUNK)
            selected = torch.gather(
                logits_row, -1, index_row.unsqueeze(-1)
            ).squeeze(-1).float()
            token_logprobs.append((selected - lse_row).to(logits.dtype))
        return torch.stack(token_logprobs)


# =============================================================================
# Chunked token entropy — ported from old implementation
# =============================================================================

def chunked_token_entropy(
    logits_2d:  torch.Tensor,          # (T, V)
    chunk_size: int = ENTROPY_CHUNK,
) -> torch.Tensor:
    """
    Exact token entropy H(p) = logsumexp(z) - E_p[z], computed in vocab chunks.
    Avoids materializing full softmax/log-softmax tensors.
    """
    T      = logits_2d.size(0)
    device = logits_2d.device

    m = torch.full((T,), float("-inf"), device=device, dtype=torch.float32)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc = logits_2d[:, s:s + chunk_size].float()
        m  = torch.maximum(m, zc.max(dim=-1).values)

    sum_exp  = torch.zeros_like(m)
    sum_zexp = torch.zeros_like(m)
    for s in range(0, logits_2d.size(-1), chunk_size):
        zc        = logits_2d[:, s:s + chunk_size].float()
        wc        = torch.exp(zc - m.unsqueeze(-1))
        sum_exp   = sum_exp  + wc.sum(dim=-1)
        sum_zexp  = sum_zexp + (wc * zc).sum(dim=-1)

    lse        = m + torch.log(sum_exp)
    expected_z = sum_zexp / sum_exp
    return (lse - expected_z).to(logits_2d.dtype)
