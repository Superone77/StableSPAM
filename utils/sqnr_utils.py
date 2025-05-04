import math, torch
from torch.utils.tensorboard import SummaryWriter
from utils.fake_quantization import _quantize_tensor   # 复用现成的权重量化
LOG_EPS = 1e-12

def calc_sqnr(x, x_hat):
    mse = (x - x_hat).pow(2).mean()
    power = x.pow(2).mean()
    return 10 * torch.log10(power / (mse + LOG_EPS))

def quant_dequant(tensor, n_bits, group_sz):
    q, s, z = _quantize_tensor(tensor, q_group_size=group_sz, n_bit=n_bits)
    return (q.to(tensor.device) - z) * s      # 反量化

class SQNRLogger:
    def __init__(self, model, log_every, log_dir,writer):
        self.model = model
        self.log_every = log_every
        self.writer = writer
        self._step = 0
        self._handles = self._register_act_hooks()

    # -------- 权重 SQNR，每 step 调一次 ----------
    @torch.no_grad()
    def log_weight_sqnr(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "weight"):           # 兼容 QLinear 和普通 Linear
                w = module.weight
                bits = getattr(module, "num_bits", 4)
                gsz  = getattr(module, "group_size", -1)
                w_hat = quant_dequant(w, bits, gsz)
                sqnr = calc_sqnr(w, w_hat)
                self.writer.add_scalar(f"sqnr/weight/{name}", sqnr.item(), self._step)

    # -------- 激活 SQNR，利用 forward_hook ----------
    def _register_act_hooks(self):
        handles = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                bits = getattr(mod, "num_bits", 4)
                gsz  = getattr(mod, "group_size", -1)
                def _hook(m, inp, outp, lname=name, nbits=bits, g=gsz):
                    x = inp[0].detach()
                    x_hat = quant_dequant(x, nbits, g)
                    sqnr = calc_sqnr(x, x_hat)
                    self.writer.add_scalar(f"sqnr/act/{lname}", sqnr.item(), self._step)
                handles.append(mod.register_forward_hook(_hook))
        return handles

    def step(self):
        self._step += 1
        if self._step % self.log_every == 0:
            self.log_weight_sqnr()

    def close(self):
        for h in self._handles:
            h.remove()
        self.writer.close()