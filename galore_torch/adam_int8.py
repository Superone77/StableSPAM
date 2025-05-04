# Copyright (c) Qualcomm AI Research.
# SPDX‑License‑Identifier: BSD‑3‑Clause‑Clear

from bitsandbytes.optim.optimizer import Optimizer2State

import torch
from .utils import calc_sqnr


class Adam8bitSQNR(Optimizer2State):
    """
    Adam‑8bit 优化器的 SQNR 仿真版：
      • 每隔 sqnr_update_gap 步，反量化 m / v 并计算 SQNR
      • 兼容 paged‑optimizer & block‑wise 压缩
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        optim_bits: int = 32,
        args=None,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
        sqnr_update_gap: int = 50,
        update_proj_gap: int = 1000,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged,
        )
        self.sqnr_update_gap = sqnr_update_gap
        self._step_id = 0
        self.update_proj_gap = update_proj_gap
        self.betas = betas

    # --------------------------------------------------------------------- #
    #                               helpers                                 #
    # --------------------------------------------------------------------- #
    def _dequant(
        self,
        q: torch.Tensor,
        absmax: torch.Tensor,
        signed: bool = True,
        block_wise: bool = False,
        blocksize: int = 256,
    ) -> torch.Tensor:
        """
        将 uint8 / int8 压缩状态张量反量化成 float32。

        Args:
            q       : uint8 / int8 tensor.
            absmax  : 每 block / per‑tensor 的 abs‑max（float32）。
        Returns:
            float32 tensor 与原量化前大小一致。
        """
        if block_wise:
            # 假设最后一维是连续 block，把 absmax 展平匹配每个元素
            absmax = absmax.repeat_interleave(blocksize).view(*q.shape)
        else:
            absmax = absmax.view(*([1] * (q.ndim - 1)))  # broadcast

        if signed:
            return (q.float() - 128.0) * (absmax / 127.0)
        return q.float() * (absmax / 255.0)

    # --------------------------------------------------------------------- #
    #                               training                                #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step and（可选）计算 SQNR。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_id += 1

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # fairseq 纯 FP16 训练需要
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]

                # -------- state 初始化 -------- #
                if "step" not in state:
                    state["step"] = 0
                    state["sqnr_m"] = torch.tensor(0.0, device=p.grad.device)
                    state["sqnr_v"] = torch.tensor(0.0, device=p.grad.device)

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                # -------- 前一时刻的 m / v（反量化） -------- #
                if self._step_id % self.sqnr_update_gap == 0:
                    # m_prev
                    if "state1" in state and "absmax1" in state:
                        signed = state["state1"].dtype is torch.int8
                        m_prev = self._dequant(
                            state["state1"], state["absmax1"],
                            signed=signed, block_wise=True
                        )

                        # v_prev
                        signed = state["state2"].dtype is torch.int8
                        v_prev = self._dequant(
                            state["state2"], state["absmax2"],
                            signed=signed, block_wise=True
                        )
                    else:
                        m_prev = torch.zeros_like(p.data)
                        v_prev = torch.zeros_like(p.data)

                    # -------- 真实（全精度）Adam 更新 -------- #
                    grad = p.grad.detach()
                    m_true = m_prev.mul(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
                    v_true = v_prev.mul(self.betas[1]).addcmul_(
                        grad, grad, value=1 - self.betas[1]
                    )
                    state["m_true"] = m_true
                    state["v_true"] = v_true

                # 原 Optimizer2State 的核心 update
                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)

                # -------- SQNR 计算 -------- #
                if self._step_id % self.sqnr_update_gap == 0:
                    signed = state["state1"].dtype is torch.int8
                    m_hat = self._dequant(
                        state["state1"], state["absmax1"],
                        signed=signed, block_wise=True
                    )
                    signed = state["state2"].dtype is torch.int8
                    v_hat = self._dequant(
                        state["state2"], state["absmax2"],
                        signed=signed, block_wise=True
                    )

                    state["sqnr_m"] = calc_sqnr(m_true, m_hat)
                    state["sqnr_v"] = calc_sqnr(v_true, v_hat)

        torch.cuda.synchronize()

        if self.is_paged:
            # paged 操作异步，确保状态一致
            torch.cuda.synchronize()

        return loss
