# Copyright (c) 2025  Wanqi Yang & contributors
# MIT License
import math, torch
from bitsandbytes.optim.optimizer import Optimizer2State
from torch.optim.optimizer import Optimizer
from .stablespam import StableSPAM
from .utils import calc_sqnr

    

# ------------- ① 余温保持的 Cosine Decay（直接拷自 stablespam.py） --------------
class _CosineDecay:
    def __init__(self, death_rate, T_max, eta_min=0.5):
        dummy = torch.nn.Parameter(torch.zeros(1))
        self._sgd = torch.optim.SGD([dummy], lr=death_rate)
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._sgd, T_max + 1, eta_min
        )

    def __call__(self, step: int):
        self._sched.step(step)
        return self._sgd.param_groups[0]["lr"]  # scaling factor



# ------------- ② Stable‑SPAM‑8bit Optimizer ------------------
class StableSPAM8bit(Optimizer2State):             
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        gamma1=0.7,
        gamma2=0.9,
        gamma3=0.999,
        total_T=None,
        eta_min=0.5,
        update_proj_gap=1000,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        sqnr_update_gap = 50
    ):
        # ! 调用 Optimizer2State.__init__ 会自动创建 8‑bit 状态
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits=8,            # ← 指定 8‑bit
            args=None,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )
        # Stable‑SPAM 专属超参
        self.g1, self.g2, self.th = gamma1, gamma2, gamma3
        self.total_T = total_T
        self.update_proj_gap = update_proj_gap
        self._step_id = 0
        self._warm = _CosineDecay(1.0, total_T, eta_min) if total_T else None
        self.eps = eps
        self.betas = betas
        self.sqnr_update_gap = sqnr_update_gap

    def _dequant(self, q, absmax, signed=True, block_wise=False, blocksize=256):
        """
        q: uint8 Tensor, absmax: float32 Tensor
        return: float32 Tensor
        """
        if block_wise:
            # 把 absmax 展平复制到每个元素（假设 last dim contiguous block）
            expand_shape = [absmax.numel() * blocksize]
            absmax = absmax.repeat_interleave(blocksize)[:q.numel()].view_as(q)
        else:
            absmax = absmax.view([-1] + [1]*(q.ndim-1))  # broadcast
        if signed:
            return (q.float() - 128.0) * (absmax / 127.0)
        else:
            return q.float() * (absmax / 255.0)


    # ---------- ③ 核心：clip + norm scaling + momentum reset ----------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_id += 1
        scale = self._warm(self._step_id) if self._warm else 1.0

        # ----------- 遍历 param group ----------
        grad_norm_ori = 0
        grad_norm_after_clipping = 0
        grad_norm_after_clipping_and_normalization = 0
        for gidx, group in enumerate(self.param_groups):
            for pidx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                grad_norm_ori+=torch.norm(grad).item()

                # === 1) Weight Decay on weight (AdamW style) ===
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # === 2) Spike‑Aware clipping (同原版 stablespam.py) ===
                g_abs = grad.abs()
                g_max = g_abs.max()
                state = self.state[p]
                if "m_max_t" not in state:
                    state.update(
                        {
                            "m_max_t": torch.zeros_like(g_max),
                            "m_norm": torch.tensor(0.0, device=grad.device),
                            "v_norm": torch.tensor(0.0, device=grad.device),
                            'sqnr_m':torch.tensor(0.0, device=grad.device),
                            'sqnr_v':torch.tensor(0.0, device=grad.device),
                        }
                    )

                state["m_max_t"].mul_(self.th).add_(g_max, alpha=1 - self.th)
                g_hat = state["m_max_t"] / (1 - self.th ** self._step_id)
                mask = g_abs > g_hat
                if mask.any():
                    grad[mask] = grad[mask] / g_max * g_hat
                grad_norm_after_clipping+=torch.norm(grad).item()
                
                # === 3) Norm scaling (γ1, γ2 version) ============
                g_norm = grad.norm()
                m_norm = state["m_norm"]
                v_norm = state["v_norm"]

                m_norm.mul_(self.g1 * scale).add_(g_norm, alpha=1 - self.g1 * scale)
                v_norm.mul_(self.g2).add_(g_norm ** 2, alpha=1 - self.g2)

                m_hat = m_norm / (1 - (self.g1 * scale) ** self._step_id)
                v_hat = v_norm / (1 - self.g2 ** self._step_id)
                c_t = m_hat / (v_hat.sqrt() + self.eps)
                if g_norm > 0:
                    grad.mul_(c_t / g_norm)

                state["m_norm"], state["v_norm"] = m_norm, v_norm
                
                
                # === 4) 把处理后的梯度写回，再调用 8‑bit 核心更新 ===
                #### debug ######
                p.grad = grad
                #################
                grad_norm_after_clipping_and_normalization+=torch.norm(grad).item()
                # === 5) Momentum reset (exp_avg/exp_avg_sq → 0) ===
                if self.update_proj_gap > 0 and self._step_id % self.update_proj_gap == 0:
                    # state["state1"].zero_()
                    # state["state2"].zero_()
                    # state["step"] = 1  # 对应 Adam 的 time step
                    self.init_state(group, p, gidx, pidx)
                    
                if "step" not in state:        # 初始化 8‑bit 状态
                    self.init_state(group, p, gidx, pidx)

                if "state1" in state and 'absmax1' in state and self._step_id % self.sqnr_update_gap == 0:
                    if state['state1'].dtype == torch.uint8:
                        signed = False
                    elif state['state1'].dtype == torch.int8:
                        signed = True
                    else:
                        print(f"state1 dtype error with {state['state1'].dtype}")
                    m_prev = self._dequant(state['state1'],
                                           state['absmax1'],
                                           signed=signed,
                                           block_wise=True)
                    if state['state2'].dtype == torch.uint8:
                        signed = False
                    elif state['state2'].dtype == torch.int8:
                        signed = True
                    else:
                        print(f"state2 dtype error with {state['state2'].dtype}")
                    v_prev = self._dequant(state['state2'],
                                           state['absmax2'],
                                           signed=signed,
                                            block_wise=True)
                else:
                    m_prev = torch.zeros_like(p.data)
                    v_prev = torch.zeros_like(p.data)
                m_true = m_prev.mul(self.betas[0]).add(grad, alpha = 1-self.betas[0])
                v_true = v_prev.mul(self.betas[1]).addcmul_(grad, grad, value=1-self.betas[1])
                state['m_true'] = m_true
                state['v_true'] = v_true
                
                self.prefetch_state(p)
                self.update_step(group, p, gidx, pidx)   # ← bitsandbytes 内核

                # sqnr compute
                if "state1" in state and 'absmax1' in state and self._step_id % self.sqnr_update_gap == 0:
                    if state['state1'].dtype == torch.uint8:
                        signed = False
                    elif state['state1'].dtype == torch.int8:
                        signed = True
                    else:
                        print(f"state1 dtype error with {state['state1'].dtype}")
                    m_hat = self._dequant(state['state1'],
                                            state['absmax1'],
                                            signed=signed,
                                            block_wise=True)
                    if state['state2'].dtype == torch.uint8:
                        signed = False
                    elif state['state2'].dtype == torch.int8:
                        signed = True
                    else:
                        print(f"state2 dtype error with {state['state2'].dtype}")
                    v_hat = self._dequant(state['state2'],
                                            state['absmax2'],
                                            signed=signed,
                                            block_wise=True)
                    state['sqnr_m'] = calc_sqnr(m_true, m_hat)
                    state['sqnr_v'] = calc_sqnr(v_true, v_hat)
            
        # print(grad_norm_ori, grad_norm_after_clipping, grad_norm_after_clipping_and_normalization)
        torch.cuda.synchronize()
        return loss, torch.tensor(grad_norm_after_clipping_and_normalization).sqrt()