import torch
from torch import nn
from math import sqrt
from typing import Optional, Tuple
import torch.nn.functional as F

class ScaledLayerNorm(nn.Module):
    """
    Wrap a LayerNorm (or RMSNorm) and scale its output by 1/sqrt(layer_idx+1).
    """
    def __init__(self, orig_ln: nn.Module, layer_idx: int):
        super().__init__()
        self.orig_ln = orig_ln
        # ℓ 从 0 开始，scale = 1 / sqrt(ℓ+1)
        self.scale = 1.0 / sqrt(layer_idx + 1)

    def forward(self, x):
        return self.orig_ln(x) * self.scale

class ScaledSwiglu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delayed = False
        self.initialized = False
        self.scale = None
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.chunk(x, 2, dim=-1)

        if self.delayed:
            tmp = x[1].detach().abs().max(dim=-1, keepdim=True)[0]
            if self.initialized:
                s = self.scale.clone()
                self.scale.zero_()
            else:
                s = tmp
                self.scale = torch.zeros_like(tmp)
                self.initialized = True
            self.scale.add_(tmp)
        else:
            s = x[1].abs().max(dim=-1, keepdim=True)[0]
            # if os.getenv('DETACH_SCALED_SWIGLU', 'false').lower() == 'true':
            #     s = x[1].detach().abs().max(dim=-1, keepdim=True)[0]
            # else:
            #     s = x[1].abs().max(dim=-1, keepdim=True)[0]
        
        tmp = x[1] / s
        return F.silu(x[0]) * tmp, s


