"""TopK sparse autoencoder (Gao et al., 2024) used by both SAE versions of contextual EMI.

Inputs are centered on the training mean and divided by one scalar so the average squared norm per dimension is 1;
encode() and decode() take and return vectors in the model's own space. The encoder keeps exactly the k largest
pre-activations (then ReLU), the decoder rows have unit norm, and dead features are revived with the AuxK loss
(the residual reconstructed by the top dead features, weight 1/32).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_COEFFICIENT = 1.0 / 32


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int, aux_k: int | None = None):
        super().__init__()
        self.d_in, self.n_features, self.k = d_in, n_features, k
        self.aux_k = aux_k or min(512, max(1, n_features // 2))
        decoder = F.normalize(torch.randn(n_features, d_in), dim=1)
        self.W_dec = nn.Parameter(decoder)
        self.W_enc = nn.Parameter(decoder.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("in_mean", torch.zeros(d_in))
        self.register_buffer("in_scale", torch.ones(()))

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, sample: torch.Tensor) -> None:
        mean = mean.to(self.in_mean.device, torch.float32)
        sample = sample.to(self.in_mean.device, torch.float32)
        self.in_mean.copy_(mean)
        self.in_scale.copy_(((sample - mean) ** 2).sum(dim=1).mean().div(self.d_in).sqrt().clamp(min=1e-8))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.in_mean) / self.in_scale

    def _pre(self, x_n: torch.Tensor) -> torch.Tensor:
        return F.relu((x_n - self.b_dec) @ self.W_enc + self.b_enc)

    def _topk(self, pre: torch.Tensor) -> torch.Tensor:
        values, indices = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, indices, values)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._topk(self._pre(self._normalize(x)))

    def _decode_normalized(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self._decode_normalized(f) * self.in_scale + self.in_mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor, dead: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Normalized MSE (fraction of variance unexplained in the batch) + AuxK on dead features."""
        x_n = self._normalize(x)
        pre = self._pre(x_n)
        f = self._topk(pre)
        x_hat = self._decode_normalized(f)
        variance = ((x_n - x_n.mean(dim=0)) ** 2).sum(dim=-1).mean().clamp(min=1e-8)
        mse = ((x_hat - x_n) ** 2).sum(dim=-1).mean() / variance
        aux = torch.zeros((), device=x.device)
        dead = dead.to(x.device)
        if dead.any():
            k_aux = min(self.aux_k, int(dead.sum()))
            residual = (x_n - x_hat).detach()
            masked = pre.masked_fill(~dead, float("-inf"))
            values, indices = masked.topk(k_aux, dim=-1)
            f_aux = torch.zeros_like(pre).scatter_(-1, indices, values.clamp(min=0))
            aux = ((f_aux @ self.W_dec - residual) ** 2).sum(dim=-1).mean() / (residual ** 2).sum(dim=-1).mean().clamp(min=1e-8)
        return mse + AUX_COEFFICIENT * aux, {"fvu": float(mse.detach()), "aux": float(aux.detach()), "fired": (f > 0).any(dim=0)}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def fvu(self, x: torch.Tensor, batch: int = 8192) -> float:
        """Fraction of variance unexplained over x, in the model's space."""
        error, total = 0.0, 0.0
        mean = x.float().mean(dim=0)
        for start in range(0, len(x), batch):
            chunk = x[start : start + batch].float()
            x_hat, _ = self(chunk)
            error += float(((x_hat - chunk) ** 2).sum())
            total += float(((chunk - mean) ** 2).sum())
        return error / max(total, 1e-12)

    def save(self, path: str | Path, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "config": {"d_in": self.d_in, "n_features": self.n_features, "k": self.k, "aux_k": self.aux_k},
                    "meta": {**meta, "name": path.parent.name}}, path)


def load_sae(path: str | Path, device) -> tuple[TopKSAE, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sae = TopKSAE(**checkpoint["config"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.to(device).eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae, checkpoint["meta"]
