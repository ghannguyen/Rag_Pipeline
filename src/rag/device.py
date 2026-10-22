"""
Torch device selection shared by embedding, indexing and re-ranking.

Priority: TORCH_DEVICE env override -> CUDA -> CPU.

Apple MPS is opt-in (TORCH_DEVICE=mps): it shares RAM with the OS, and on an
8 GB M1 keeping bge-m3 + bge-reranker resident on MPS pushed the server into
swap and made chat ~2.3x slower than CPU.
"""

import os


def pick_device() -> str:
    override = os.getenv("TORCH_DEVICE", "").strip().lower()
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
