import torch
import torch.nn as nn
from .utils_for_quant import OPTIMAL_GAUSSIAN_SCALES

class StochasticRoundingQuantizer(nn.Module):
    def __init__(self, bits=4, centered=True):
        super().__init__()
        self.bits = bits
        self.centered = centered
        self.n_levels = 2 ** bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            r = x_clip / step + 0.5
            base = torch.floor(r)
            prob = r - base
            rnd = torch.rand_like(prob)
            q = (base + (rnd < prob).float()) * step - 0.5 * step
        else:
            neg_scale = -scale * (self.n_levels - 2) / self.n_levels
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            r = x_clip / step
            base = torch.floor(r)
            prob = r - base
            rnd = torch.rand_like(prob)
            q = (base + (rnd < prob).float()) * step
        return x + (q - x).detach()

    def re_randomize(self):
        pass
