"""Memory-safe chunked frequency scoring for beam search agents."""

from __future__ import annotations

import torch


def chunked_freq_score(
    model: torch.nn.Module,
    ref_sample: torch.Tensor,
    cand_embs: torch.Tensor,
    sampled_fc: torch.Tensor | None = None,
    max_strength: float = 0.0,
) -> torch.Tensor:
    """Score candidates against references in memory-safe chunks.

    Chunks along the candidate (N) dimension so peak GPU memory stays
    bounded regardless of how many candidates a step produces.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)`` — frequency count (not normalised) per candidate.
    """
    N = cand_embs.shape[0]
    m = ref_sample.shape[0]
    device = cand_embs.device

    # Peak per chunk: violations(C,m,4) + preds(C,m,8) + is_super(C,m,1) ≈ C*m*13
    cand_chunk = max(1, (512 * 1024 * 1024) // max(1, m * 13))

    freq_counts = torch.empty(N, device=device)

    with torch.no_grad():
        for cs in range(0, N, cand_chunk):
            ce = min(cs + cand_chunk, N)
            violations = model.batch_predict(ref_sample, cand_embs[cs:ce])
            preds = model.clf_model(violations.unsqueeze(-1))
            is_super = torch.argmax(preds, dim=-1).bool()
            chunk_freq = is_super.float().sum(dim=1)

            if sampled_fc is not None:
                masked_fc = is_super.float() * sampled_fc.unsqueeze(0)
                lower_bounds = masked_fc.max(dim=1).values
                non_anom = lower_bounds > max_strength
                for i in non_anom.nonzero(as_tuple=True)[0].tolist():
                    chunk_freq[i] = lower_bounds[i].item() * m

            freq_counts[cs:ce] = chunk_freq
            del violations, preds, is_super

    return freq_counts
