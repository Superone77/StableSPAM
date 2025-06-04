import math
import torch
from torch.optim import Optimizer
from .utils import calc_sqnr
from .fp4 import _QuantDequantFP8, _QuantDequantFP4
# FP8 最大值（参考 IEEE754 定义）
_FP8_E4M3_MAX = 240.0
_FP8_E5M2_MAX = 57344.0


class FP8Adam(Optimizer):
    """
    在标准 Adam 基础上，对 weight/m1/m2 三组持久变量进行 FP8 量化模拟。

    Args:
        params: 待优化参数
        lr: 学习率
        betas: Adam 两个动量因子 (beta1, beta2)
        eps: 数值稳定项
        weight_decay: L2 正则化系数
        dtype: 'e4m3' or 'e5m2'
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 dtype: str = 'e4m3',
                 sqnr_update_gap = 50,
                 update_proj_gap=1000):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("Invalid betas: {}".format(betas))

        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # 针对 weight, m1, m2 三种持久态的量化器
        self.q_m1 = _QuantDequantFP8(dtype)
        self.q_m2 = _QuantDequantFP8(dtype)
        self.update_proj_gap = update_proj_gap
        self.sqnr_update_gap = sqnr_update_gap
        self._step_id = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        self._step_id += 1
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("FP8Adam 不支持稀疏梯度")

                state = self.state[p]
                # 初始化 state
                if len(state) == 0:
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['sqnr_m'] = torch.tensor(0.0, device=p.grad.device)
                    state['sqnr_v'] = torch.tensor(0.0, device=p.grad.device)
                if self.update_proj_gap > 0 and self._step_id % self.update_proj_gap == 0:
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state['step'] += 1
                t = state['step']

                # Adam 更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # 权重衰减（L2）
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # 最终参数更新
                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                p.data.add_(update, alpha=-step_size)

                # —— 以下三行，在更新后对持久化变量做 FP8 量化模拟 ——
                state["exp_avg"]    = self.q_m1(exp_avg)
                state["exp_avg_sq"]= self.q_m2(exp_avg_sq)
                if self._step_id % self.sqnr_update_gap == 0:
                    state['sqnr_m'] = calc_sqnr(exp_avg, state["exp_avg"])
                    state['sqnr_v'] = calc_sqnr(exp_avg_sq, state["exp_avg_sq"])

        return loss


class FP4Adam(Optimizer):
    """
    在标准 Adam 基础上，对 weight/m1/m2 三组持久变量进行 FP8 量化模拟。

    Args:
        params: 待优化参数
        lr: 学习率
        betas: Adam 两个动量因子 (beta1, beta2)
        eps: 数值稳定项
        weight_decay: L2 正则化系数
        dtype: 'e4m3' or 'e5m2'
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 block_size: int = 32, 
                 scale_fmt: str = 'e4m3',
                 sqnr_update_gap = 50,
                 update_proj_gap=1000):
        if not 0.0 <= lr:
            raise ValueError("Invalid lr: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("Invalid betas: {}".format(betas))

        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # 针对 weight, m1, m2 三种持久态的量化器
        self.q_m1 = _QuantDequantFP4(block_size, scale_fmt)
        self.q_m2 = _QuantDequantFP4(block_size, scale_fmt)
        self.update_proj_gap = update_proj_gap
        self.sqnr_update_gap = sqnr_update_gap
        self._step_id = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        self._step_id += 1
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("FP8Adam 不支持稀疏梯度")

                state = self.state[p]
                # 初始化 state
                if len(state) == 0:
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['sqnr_m'] = torch.tensor(0.0, device=p.grad.device)
                    state['sqnr_v'] = torch.tensor(0.0, device=p.grad.device)
                if self.update_proj_gap > 0 and self._step_id % self.update_proj_gap == 0:
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state['step'] += 1
                t = state['step']

                # Adam 更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # 权重衰减（L2）
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # 最终参数更新
                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                p.data.add_(update, alpha=-step_size)

                # —— 以下三行，在更新后对持久化变量做 FP8 量化模拟 ——
                state["exp_avg"]    = self.q_m1(exp_avg)
                state["exp_avg_sq"]= self.q_m2(exp_avg_sq)
                if self._step_id % self.sqnr_update_gap == 0:
                    state['sqnr_m'] = calc_sqnr(exp_avg, state["exp_avg"])
                    state['sqnr_v'] = calc_sqnr(exp_avg_sq, state["exp_avg_sq"])

        return loss
