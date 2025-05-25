import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

import sys
from .hadamard_transformer_helper import naive_hadamard_transform_with_scale as hadamard_transform
from .sqnr_utils import calc_sqnr

class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x


class UniformQuantizer(BaseQuantizer):
    def forward(self, x):
        if not self.training:
            return x
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True) + 1e-8
        step = scale * 2 / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()


class ClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = (
                (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
            ).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardTrustQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(32, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True)
        self.matrix = None
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 32),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class TrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HadamardTrustQuantizer(TrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class GaussianSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.register_buffer("levels", self._compute_gaussian_levels())

    def _compute_gaussian_levels(self):
        levels = np.linspace(-3, 3, self.n_levels)
        boundaries = np.zeros(self.n_levels + 1)

        for _ in range(20):
            boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
            boundaries[0] = -float("inf")
            boundaries[-1] = float("inf")

            new_levels = []
            for i in range(self.n_levels):
                b_left, b_right = boundaries[i], boundaries[i + 1]

                def f(x):
                    return x * norm.pdf(x)

                integral_num = integrate.quad(f, b_left, b_right)[0]
                integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
                if integral_den > 1e-10:
                    new_levels.append(integral_num / integral_den)
                else:
                    new_levels.append(levels[i])
            levels = np.array(new_levels)
        return torch.tensor(levels, dtype=torch.float32)

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        return x + (xq - x).detach()


class GaussianClipQuantizer(GaussianSTEQuantizer):
    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (x_norm.abs() <= self.levels[-1]).float()
        return x * mask + (xq - x * mask).detach()


class GaussianTrustQuantizer(GaussianSTEQuantizer):
    def __init__(self, bits=4, trust=None):
        super().__init__(bits)
        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            xq = xq @ self.matrix.T
            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


FP4_LEVELS = [
    -2.92247856,
    -1.94831904,
    -1.46123928,
    -0.97415952,
    -0.73061964,
    -0.48707976,
    -0.24353988,
    0.0,
    0.0,
    0.24353988,
    0.48707976,
    0.73061964,
    0.97415952,
    1.46123928,
    1.94831904,
    2.92247856,
]


class FP4STEQuantizer(GaussianSTEQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class FP4ClipQuantizer(GaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class FP4TrustQuantizer(GaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4ClipQuantizer(HalfHadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HadamardFP4ClipQuantizer(HadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4TrustQuantizer(HalfHadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class HadamardFP4TrustQuantizer(HadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class FourEightMaskedQuantizer(BaseQuantizer):
    def __init__(self, p=2.0):
        super().__init__(16)
        self.p = p

    def forward(self, x):
        x_reshaped = x.reshape(-1, 4, 2)
        _, idx = x_reshaped.norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        mask = torch.ones_like(x_reshaped, dtype=torch.bool)
        mask[torch.arange(x_reshaped.size(0)).repeat(2, 1).T, idx, :] = False
        mask = mask.reshape(x.shape).float()

        return x * mask


class FourEightSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, p: float = 2.0):
        super().__init__(bits)
        self.p = p

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        return x + (xq - x).detach()


class FourEightClipQuantizer(FourEightSTEQuantizer):
    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(x) <= scale).float()
        return x * mask + (xq - x * mask).detach()


class FourEightTrustQuantizer(FourEightSTEQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, p)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


# torch._dynamo.config.optimize_ddp=False # uncommend if actually using ErfClipQuantizer
class ErfFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xq, buffer, mask):
        ctx.save_for_backward(buffer, mask)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        buffer, mask = ctx.saved_tensors
        mask = mask.float()

        return (
            (grad_output + buffer) * mask,
            None,
            grad_output * (1 - mask) - buffer * mask,
            None,
        )


class ErfClipQuantizer(ClipQuantizer):
    def __init__(self, bits=4, acc_dtype=torch.float32):
        super().__init__(bits, True)
        self.acc_dtype = acc_dtype
        self.register_parameter("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.acc is None:
                self.acc = nn.Parameter(
                    torch.zeros_like(x, dtype=self.acc_dtype), requires_grad=True
                )
            elif self.acc.grad is not None:
                self.acc.data += self.acc.grad
                self.acc.grad = None

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        mask = (torch.abs(x) <= scale).float()

        return ErfFn().apply(x, xq, self.acc, mask)


class FlushAccFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, acc):
        ctx.save_for_backward(acc)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (acc,) = ctx.saved_tensors
        return grad_output + acc, None


class ClipAccQuantizer(STEQuantizer):
    def __init__(
        self,
        bits=4,
        centered=True,
        flush_every: int = 64,
        acc_dtype=torch.float32,
        scale: float = None,
    ):
        super().__init__(bits, centered)

        if scale is None:
            scale = 1 / flush_every

        self.acc_dtype = acc_dtype
        self.flush_every = flush_every
        self.counter = 0
        self.scale = scale
        self.register_buffer("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.counter == 0:
                if self.acc is None:
                    self.acc = torch.zeros_like(x, dtype=self.acc_dtype)
                else:
                    self.acc.data = torch.zeros_like(x, dtype=self.acc_dtype)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = ((neg_scale <= x) & (x <= scale)).float()

        self.counter += 1
        if self.counter == self.flush_every:
            self.counter = 0
            grad_flow_output = FlushAccFn().apply(
                x * mask + x * (1 - mask) * self.scale,
                (self.acc * self.scale).to(x.dtype),
            )
        else:
            grad_flow_output = x * mask + self.acc * (1 - mask)

        return grad_flow_output + (xq - grad_flow_output).detach()


class LSQQuantizer(nn.Module):
    """
    Implementation of LSQ quantizer from https://arxiv.org/abs/1902.08153
    LSQ uses a learnable step size for quantization. This learnable step size(alpha) is initialized using the optimal gaussian scale
    ans must be normalized with a weight decay.
    """

    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__()
        # NOTE: raise_zero should never be used with FP quantization

        self.bits = bits
        self.n_levels = 2**bits
        self.all_positive = all_positive
        self.raise_zero = raise_zero

        self.q_min, self.q_max = self.get_dtype_bounds()

        self.is_alpha_init = False
        self.alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def get_dtype_bounds(self):
        if not self.all_positive:
            q_min = -self.n_levels / 2
            q_max = self.n_levels / 2 - 1
        else:
            q_min = 0
            q_max = self.n_levels - 1
        return q_min, q_max

    def cast(self, x):
        # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
        # NOTE: raise_zero should never be used with FP quantization
        return x.round()

    def ste_cast(self, x):
        return (self.cast(x) - x).detach() + x

    def grad_scale(self, x, scale):
        return (x - x * scale).detach() + x * scale

    @torch.no_grad()
    def get_initial_step_value(self, x):
        return (
            torch.mean(torch.abs(x.detach())) * 2 / (np.sqrt(self.q_max))
        )  # LSQ initialization

    def get_learnable_step(self, x):
        if not self.is_alpha_init:
            with torch.no_grad():
                step = self.get_initial_step_value(x)
                self.alpha_weight.data.multiply_(
                    torch.tensor(
                        step,
                        dtype=self.alpha_weight.dtype,
                        device=self.alpha_weight.device,
                    )
                )
            self.is_alpha_init = True
        return self.alpha_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step

        return xq + step * 1e-9  # extra term to ensure gradient flow


class LSQPlusWeightQuantizer(LSQQuantizer):
    @torch.no_grad()
    def get_initial_step_value(self, x):
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x**2)) + 1e-8
        step = 2 * scale / (self.n_levels - 1)
        return step


class LSQPlusActivationQuantizer(LSQPlusWeightQuantizer):
    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__(bits, raise_zero, all_positive, **kwargs)
        self.beta_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.is_beta_init = False

    @torch.no_grad()
    def get_initial_bias_value(self, x):
        return x.min() - self.alpha_weight * self.q_min

    def get_learnable_bias(self, x):
        if not self.is_beta_init:
            with torch.no_grad():
                bias = self.get_initial_bias_value(x)
                self.beta_weight.data.add_(
                    torch.tensor(
                        bias,
                        dtype=self.beta_weight.dtype,
                        device=self.beta_weight.device,
                    )
                )
            self.is_beta_init = True
        return self.beta_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        bias = self.get_learnable_bias(x)
        bias = self.grad_scale(bias, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = (x - bias) / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step + bias
        return xq + step * 1e-9  # extra term to ensure gradient flow


class PACTQuantizer(LSQQuantizer):
    """
    Implementation of PACT quantizer from https://arxiv.org/abs/1805.06085
    PACT and LSQ are quite similar and do the same thing for forward pass.
    The difference is in the backward pass where PACT does not perform a full gradient flow.
    """

    def forward(self, x):
        step = self.get_learnable_step(x)
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs - 1 / 2)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs)
            xscr = self.ste_cast(xsc)
        xq = xscr * step
        xq = xq * clamp_mask + (xq - xq * clamp_mask).detach()
        return xq + step * 1e-9  # extra term to ensure gradient flow


QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    "HadamardClipQuantizer": HadamardClipQuantizer,
    "TrustQuantizer": TrustQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "HadamardTrustQuantizer": HadamardTrustQuantizer,
    "GaussianSTEQuantizer": GaussianSTEQuantizer,
    "GaussianClipQuantizer": GaussianClipQuantizer,
    "GaussianTrustQuantizer": GaussianTrustQuantizer,
    "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
    "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
    "HadamaardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    "FP4STEQuantizer": FP4STEQuantizer,
    "FP4ClipQuantizer": FP4ClipQuantizer,
    "FP4TrustQuantizer": FP4TrustQuantizer,
    "HalfHadamardFP4ClipQuantizer": HalfHadamardFP4ClipQuantizer,
    "HadamardFP4ClipQuantizer": HadamardFP4ClipQuantizer,
    "HalfHadamardFP4TrustQuantizer": HalfHadamardFP4TrustQuantizer,
    "HadamardFP4TrustQuantizer": HadamardFP4TrustQuantizer,
    "FourEightMaskedQuantizer": FourEightMaskedQuantizer,
    "FourEightSTEQuantizer": FourEightSTEQuantizer,
    "FourEightClipQuantizer": FourEightClipQuantizer,
    "FourEightTrustQuantizer": FourEightTrustQuantizer,
    "HalfHadamardFourEightTrustQuantizer": HalfHadamardFourEightTrustQuantizer,
    "HadamardFourEightTrustQuantizer": HadamardFourEightTrustQuantizer,
    "ErfClipQuantizer": ErfClipQuantizer,
    "ClipAccQuantizer": ClipAccQuantizer,
    "PACTQuantizer": PACTQuantizer,
    "LSQQuantizer": LSQQuantizer,
    "LSQPlusActivationQuantizer": LSQPlusActivationQuantizer,
    "LSQPlusWeightQuantizer": LSQPlusWeightQuantizer,
}


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        self.sqnr_weight = 0
        self.sqnr_act = 0

    def forward(self, x):
        x_ori = x.detach()
        x = self.activation_quantizer(x)
        self.sqnr_act = calc_sqnr(x_ori @ self.activation_quantizer.matrix,x)
        w = self.weight_quantizer(self.weight)
        self.sqnr_weight = calc_sqnr(self.weight @ self.weight_quantizer.matrix ,w)
        return F.linear(x, w, self.bias)

def prepare_model_for_quest_training_simulation_act_weight(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_quest_training_simulation_act_weight(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module:
                print('Keep in original linear layer', name, module)
                continue
            
            # NOTE(hanqing): no need to pass those stuffs
            bias_data = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            weight_data = module.weight.data
            new_layers = QuantizedLinear(in_features, out_features, bias=bias, 
                weight_quantizer=QUANTIZER_CLASSES[args.w_quant](**args.w_quant_kwargs),
                activation_quantizer=QUANTIZER_CLASSES[args.a_quant](**args.a_quant_kwargs ),
                )

            model._modules[name] = new_layers

    return model