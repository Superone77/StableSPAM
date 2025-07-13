import torch, triton, triton.language as tl

FP4_MAX   = 6.0         # fp4-(1-2-1) 动态范围上界
FP8_E4M3_MAX = 448.0# 参考原代码

# ---------------------------------------------------------------
# Triton kernel：无 scale_b_ptr，内部计算 per-block scale_b
# ---------------------------------------------------------------
@triton.jit
def nvfp4_fwd_kernel(
    x_ptr,          # *f16 / *f32  | 输入
    out_ptr,        # *f16 / *f32  | 输出
    prob_ptr,       # *f32         | [0,1) 随机数 (or 全 0)
    scale_t,        # f32          | 全局 scale_per_t
    M,              # int32        | 总元素数
    BLOCK: tl.constexpr,          # =16
    STOCHASTIC: tl.constexpr,     # bool
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < M

    # -------- 读入并预缩放 --------
    x      = tl.load(x_ptr + offs,  mask=mask, other=0.0)
    prob   = tl.load(prob_ptr + offs, mask=mask, other=0.0)
    sign   = tl.where(x >= 0.0, 1.0, -1.0)
    x_abs_scaled = tl.abs(x) / scale_t         # |x| / s_t

    # -------- 计算 per-block scale_b --------
    blk_max = tl.max(tl.where(mask, x_abs_scaled, 0.0), axis=0)
    # 避免除零 / inf
    scale_b = tl.where(blk_max > 0.0, 6.0 / blk_max, 1.0)

    y = x_abs_scaled * scale_b                # bring into [0, 6]

    # ============================================================
    # 1-2-1 fp4 量化
    # ============================================================
    if STOCHASTIC:
        # 上、下邻格
        hi = tl.where(
            y > 4, 6.0,
            tl.where(y > 3, 4.0,
                tl.where(y > 2, 3.0,
                    tl.where(y > 1.5, 2.0,
                        tl.where(y > 1.0, 1.5,
                            tl.where(y > 0.5, 1.0, 0.5)))))
        )
        lo = tl.where(
            y > 4, 4.0,
            tl.where(y > 3, 3.0,
                tl.where(y > 2, 2.0,
                    tl.where(y > 1.5, 1.5,
                        tl.where(y > 1.0, 1.0,
                            tl.where(y > 0.5, 0.5, 0.0)))))
        )
        prob_up = (y - lo) / (hi - lo + 1e-7)         # 1e-7 防除零
        q_abs_blk = tl.where(prob < prob_up, hi, lo)
    else:
        # 确定性 (阈值中心点)
        q_abs_blk = tl.where(
            y > 5,   6.0,
            tl.where(y > 3.5, 4.0,
                tl.where(y > 2.5, 3.0,
                    tl.where(y > 1.75, 2.0,
                        tl.where(y > 1.25, 1.5,
                            tl.where(y > 0.75, 1.0,
                                tl.where(y > 0.25, 0.5, 0.0))))))
        )

    # -------- 反缩放并写回 --------
    q_abs = q_abs_blk / scale_b
    q_val = sign * q_abs * scale_t
    tl.store(out_ptr + offs, q_val, mask=mask)


# ---------------------------------------------------------------
# Python 包装：无需传入 scale_per_b
# ---------------------------------------------------------------
def nvfp4_forward(
    x: torch.Tensor,
    scale_per_t: float | None = None,
    stochastic_rounding: bool = False,
):
    """
    x             : CUDA 张量 (f16/f32)，任意形状
    scale_per_t   : 若 None 则按原逻辑自动计算
    """
    
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
    fp_dtype = x.dtype
    orig_shape = x.shape
    x_flat     = x.contiguous().view(-1)
    M          = x_flat.numel()

    # ---- 全局 scale_per_t ----
    if scale_per_t is None:
        nvfp4_max = 1440.0
        scale_per_t = float(x_flat.abs().max()) / nvfp4_max
        scale_per_t = max(scale_per_t, 1e-7)          # 防 0

    # ---- 随机概率张量 ----
    if stochastic_rounding:
        prob = torch.rand_like(x_flat, dtype=torch.float32)
    else:
        prob = torch.zeros_like(x_flat, dtype=torch.float32)

    out   = torch.empty_like(x_flat)
    BLOCK = 16
    grid  = ((M + BLOCK - 1) // BLOCK,)

    nvfp4_fwd_kernel[grid](
        x_flat, out, prob,
        scale_per_t, M,
        BLOCK=BLOCK,
        STOCHASTIC=stochastic_rounding,
        num_warps=4
    )
    return out.view(orig_shape).to(fp_dtype)


# ---------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 32
    x = torch.randn(B, N, device="cuda", dtype=torch.float16) * 5

    y_det = nvfp4_forward(x, stochastic_rounding=False)
    y_sto = nvfp4_forward(x, stochastic_rounding=True)
    print("det  :", y_det[0, :8])
    print("sto  :", y_sto[0, :8])