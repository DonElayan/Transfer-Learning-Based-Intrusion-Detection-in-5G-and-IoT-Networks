"""Small encoders suitable for constrained IoT/edge devices, plus the DANN pieces."""
from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 128), out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in (*hidden, out_dim):
            # LayerNorm instead of BatchNorm: stable with tiny few-shot batches and it
            # does not carry source-domain running statistics into the target domain.
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DEncoder(nn.Module):
    """1-D counterpart of the repository's CNN-B (conv -> pool -> conv -> pool -> dense)."""

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True),
        )
        flat = 64 * ((in_dim + 3) // 4)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x.unsqueeze(1)))


def make_encoder(arch: str, in_dim: int) -> nn.Module:
    if arch == "mlp":
        return MLPEncoder(in_dim)
    if arch == "cnn1d":
        return CNN1DEncoder(in_dim)
    raise ValueError(f"unknown arch {arch!r} (mlp | cnn1d)")


class Net(nn.Module):
    """encoder + task head. head='linear' or 'mlp' (the repo's CNN-TL used a dense stack)."""

    def __init__(self, encoder: nn.Module, n_classes: int, head: str = "linear"):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        if head == "linear":
            self.head = nn.Linear(d, n_classes)
        else:
            self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(grad_reverse(z, alpha)).squeeze(-1)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
