""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from .utils import calc_sqnr

# FP8 最大值（参考 IEEE754 定义）
_FP8_E4M3_MAX = 240.0
_FP8_E5M2_MAX = 57344.0

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.5, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max+1, eta_min, last_epoch)
        self.T_max=T_max
        self.eta_min=eta_min
    def step(self,current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self,current_step):
        self.step(current_step)
        return self.sgd.param_groups[0]['lr']

import torch
from typing import Literal

# 无符号 e6m2：6 位指数（bias=31），2 位尾数 => max = 1.75 · 2**31
_FP8_E6M2_MAX = 1.75 * (2 ** 31)              # ≈ 3.758 × 10⁹

# ----------------------------------------------------------------------
#   核心类
# ----------------------------------------------------------------------
class _QuantDequantFP8:
    """
    简易 FP8 量化／反量化模拟器。
    支持
      · 'e4m3'  (torch.float8_e4m3fn)
      · 'e5m2'  (torch.float8_e5m2)
      · 'e6m2'  —— **无符号** 6‑exp / 2‑man 手动实现
    """
    _Fmt = Literal['e4m3', 'e5m2', 'e6m2']

    def __init__(self, fmt: _Fmt = 'e4m3'):
        assert fmt in ('e4m3', 'e5m2', 'e6m2','none')
        self.format = fmt

        if fmt == 'e4m3':
            self.torch_dtype = torch.float8_e4m3fn
            self.max_val    = _FP8_E4M3_MAX
        elif fmt == 'e5m2':
            self.torch_dtype = torch.float8_e5m2
            self.max_val    = _FP8_E5M2_MAX
        else:                       # 'e6m2'
            self.torch_dtype = None  # 手动量化
            self.max_val    = _FP8_E6M2_MAX
            self._bias      = (1 << 5) - 1        # 6‑bit 指数 => bias = 31
            self._frac_bits = 2                   # 尾数位数
            self._frac_scale= 1 << self._frac_bits

    # ------------------------------------------------------------------
    #   Public 入口
    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # if not x.is_floating_point() or x.dtype != torch.float32:
        #     raise ValueError("只支持 float32 输入")
        if self.format == 'none':
            return x
        if self.format == 'e6m2':
            return self._uqdq_e6m2(x)

        # -------- e4m3 / e5m2 原路径 --------
        amax = x.abs().max().item()
        if amax == 0.0:
            return x  # 全零直接返回

        scale     = self.max_val / amax
        x_scaled  = x * scale
        x_down    = x_scaled.to(self.torch_dtype)     # FP8 量化
        x_up      = x_down.to(torch.bfloat16) / scale  # 反量化
        return x_up

    # ------------------------------------------------------------------
    #   unsigned‑e6m2 专用量化‑反量化
    # ------------------------------------------------------------------
    def _uqdq_e6m2(self, x: torch.Tensor) -> torch.Tensor:
        # 负数直接截断为 0（无符号格式）
        x_pos = torch.clamp(x, min=0.0).to(torch.float32)

        amax = x_pos.max().item()
        if amax == 0.0:
            return x_pos    # 全零

        scale = self.max_val / amax
        y     = x_pos * scale             # 先 line‑scale 到 [0, max_val]

        # === 手动 FP 量化 ===
        # y = mantissa * 2**exp, mantissa ∈ [1,2)
        mant, exp = torch.frexp(y)        # mant ∈ (0.5,1]
        mant      = mant * 2.0            # → [1,2)
        exp       = exp - 1

        # 裁剪指数到 representable 区间
        exp_clp   = torch.clamp(exp, -self._bias, self._bias)

        # 2 位尾数：量化到最近的 1.xx（共 4 个刻度）
        frac      = mant - 1.0
        frac_q    = torch.round(frac * self._frac_scale) / self._frac_scale
        mant_q    = 1.0 + frac_q
        # 反构量化值
        y_q       = torch.ldexp(mant_q, exp_clp)

        # 反量化回原尺度
        x_up      = y_q / scale
        # 保持原输入的 shape & dtype
        return x_up.to(torch.bfloat16)


class StableSPAMFP8(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,gamma1=0.7,gamma2=0.9,gamma3=0.999,gamma4=0.6,total_T=None,eta_min=0.5,update_proj_gap=1000,m_dtype: str = 'e4m3',v_dtype: str = 'e6m2',sqnr_update_gap = 50):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(StableSPAMFP8, self).__init__(params, defaults)
        self.gamma1=gamma1 # 0.85 & 0.5 & 0.8,0.9
        self.gamma2=gamma2 # 0.99999 # 0.999,0.9999
        self.theta=gamma3 # 0.999 
        self.gamma4 = gamma4 #0.5 & 0.7 & 0.8
        self.total_T=total_T
        if self.total_T is not None:
            self.warmup=CosineDecay(1.0,total_T,eta_min=eta_min)   #total_T is the totoal number of update steps
        self.total_steps=0
        if self.gamma1==-1:
            self.gamma1=betas[0]
        self.update_proj_gap=update_proj_gap
        self.q_m1 = _QuantDequantFP8(m_dtype)
        # self.q_m2 = UnsignedFPQuantizer(exp_bits=5, man_bits=3)
        
        self.q_m2 = _QuantDequantFP8(v_dtype)
        
        self.sqnr_update_gap = sqnr_update_gap
        # print("hyperparameters:",gamma1,gamma2,theta,update_proj_gap)

    def __setstate__(self, state):
        super(StableSPAMFP8, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.total_steps+=1
        if self.total_T is not None:
            scale=self.warmup.get_dr(self.total_steps)
        else:
            scale=1.0
        # print("scales:",scale,self.update_proj_gap)
        grad_norm_ori = 0
        grad_norm_after_clipping = 0
        grad_norm_after_clipping_and_normalization = 0
        for group in self.param_groups:
                    
            # if "rank" in group:
            #     self.update_proj_gap=group["update_proj_gap"]

            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                grad_norm_ori+=torch.norm(grad).item()

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if "exp_avg" not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["m_norm_t"]=0
                    state["v_norm_t"]=0
                    state['m_max_t']=0
                    # state['m_min_t']=0
                    # # state["c_norm_t"]=0
                    state['sqnr_m'] = torch.tensor(0.0, device=p.grad.device)
                    state['sqnr_v'] = torch.tensor(0.0, device=p.grad.device)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                max_gradient=torch.max(grad.abs())
                # min_gradient=torch.min(grad)
                m_max_t=state["m_max_t"]
                # m_min_t=state['m_min_t']

                state['step'] += 1


                m_max_t = self.theta* m_max_t + (1 - self.theta) * max_gradient
                # m_min_t = self.theta* m_min_t + (1 - self.theta) * min_gradient
                
                m_max_hat = m_max_t / (1 - self.theta**state['step'])
                # m_min_hat = m_min_t / (1 - self.theta**state['step'])
                
                mask=grad.abs()>m_max_hat
                # mask_neg=grad<m_min_hat
                if mask.sum()>0:
                    grad[mask]=grad[mask]/max_gradient*m_max_hat
                grad_norm_after_clipping+=torch.norm(grad).item()
                state["m_max_t"]=m_max_t
                # state["m_min_t"]=m_min_t
                # ###### clipping
                grad_norm=torch.norm(grad)
                ####norm scaling
                m_norm_t,v_norm_t=state["m_norm_t"],state["v_norm_t"]
                # print("m_norm_t",m_norm_t,grad_norm)
                m_norm_t = self.gamma1 * scale*m_norm_t + (1 - self.gamma1*scale) * grad_norm
                
                v_norm_t = self.gamma2 * v_norm_t + (1 - self.gamma2) * grad_norm**2
                
                m_norm_hat = m_norm_t / (1 - (self.gamma1*scale)**state['step'])
                v_norm_hat = v_norm_t / (1 - self.gamma2**state['step'])

                c_norm_t=m_norm_hat/(torch.sqrt(v_norm_hat)+group["eps"])
                # print("grad_nrom",grad_norm,"c_norm",c_norm_t,"st",s_t,m_norm_t)
                if grad_norm>0:
                    grad=grad/grad_norm*c_norm_t
                grad_norm_after_clipping_and_normalization+=torch.norm(grad).item()
                # print(m_norm_t)
                state["m_norm_t"],state["v_norm_t"]=m_norm_t,v_norm_t

                ###############################norm scaling end#########################
                if self.update_proj_gap > 0:
                    if (self.total_steps % self.update_proj_gap == 0):
                        state["exp_avg"] = state["exp_avg"] * self.gamma4
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = state["exp_avg_sq"] * self.gamma4
                        state['step'] = int(state['step'] * self.gamma4) + 1


                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                beta1=beta1*scale

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1


                norm_grad=exp_avg/denom

                # else:
                grad=norm_grad
                p.add_(grad, alpha=-step_size)
                exp_avg_ori = exp_avg.detach()
                exp_avg_sq_ori = exp_avg_sq.detach()
                state["exp_avg"]    = self.q_m1(exp_avg)
                state["exp_avg_sq"] = self.q_m2(exp_avg_sq)
                if self.total_steps % self.sqnr_update_gap == 0:
                    state['sqnr_m'] = calc_sqnr(exp_avg_ori, state["exp_avg"])
                    state['sqnr_v'] = calc_sqnr(exp_avg_sq_ori, state["exp_avg_sq"])
                
        # print(grad_norm_ori, grad_norm_after_clipping, grad_norm_after_clipping_and_normalization)
        return loss, torch.tensor(grad_norm_after_clipping_and_normalization).sqrt()
