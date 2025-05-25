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
    

import copy, math, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def quantize_model(model, bit, group):
    """
    将全精度 model 深拷贝后 inplace 替换为 (bit, group) 配置。
    这里假设你已有 QLinear/QConv 等模块，且支持 .from_float(...)
    """
    qmodel = copy.deepcopy(model).eval()
    for name, module in qmodel.named_modules():
        for module_name, module in model.named_modules():
            if isinstance(module, QLinear):
                module.change_quant_bit(bit)
                module.change_quant_group_size(group)
            
    return qmodel

# ---------- 主评估函数 ----------
@torch.no_grad()
def eval_sqnr_vs_loss(
    fp_model,
    val_loader: DataLoader,
    criterion,
    device,
    bits=(4, 6, 8),
    groups=(32, 64, 128),
    max_batches=50,
):
    """
    返回 dict:
        results[(bit, group)] = {
            "loss": float,
            "sqnr_overall": float,
            "sqnr_per_layer": {layer_name: sqnr, ...},
        }
    """
    fp_model.eval().to(device)
    results = {}
    # 预先跑一次全精度 forward，缓存 reference 激活
    ref_acts = {}
    def save_ref(name):
        return lambda m, inp, out: ref_acts.setdefault(name, out.detach().cpu())
    hooks = [m.register_forward_hook(save_ref(n))
             for n, m in fp_model.named_modules()
             if isinstance(m, torch.nn.QLinear)]  # 只跟踪线性层，可自行扩展
    # 仅用若干 batch 即可
    loss_sum, token_sum = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        inputs, labels = (x.to(device) for x in batch)
        logits = fp_model(inputs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        token_sum += labels.size(0)
        if i == 0:  # 只需要一次 reference 激活
            for h in hooks: h.remove()
    fp_loss = loss_sum / token_sum

    # 针对不同 (bit, group) 做量化推理
    for bit in bits:
        for group in groups:
            qmodel = quantize_model(fp_model, bit, group).to(device)
            layer_sqnr = {n: [] for n, _ in qmodel.named_modules()
                          if isinstance(_, torch.nn.Linear)}

            def collect_q(name, ref):
                def fn(m, inp, out):
                    layer_sqnr[name].append(
                        compute_sqnr(ref_acts[name].to(out.device), out))
                return fn

            hooks = [m.register_forward_hook(collect_q(n, ref_acts[n]))
                     for n, m in qmodel.named_modules()
                     if n in ref_acts]

            loss_sum, token_sum = 0.0, 0
            for i, batch in enumerate(val_loader):
                if i >= max_batches: break
                inputs, labels = (x.to(device) for x in batch)
                logits = qmodel(inputs)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * labels.size(0)
                token_sum += labels.size(0)

            for h in hooks: h.remove()

            # 聚合层均值
            layer_sqnr_mean = {k: sum(v)/len(v) for k, v in layer_sqnr.items()}
            overall_sqnr = sum(layer_sqnr_mean.values()) / len(layer_sqnr_mean)

            results[(bit, group)] = dict(
                loss=loss_sum / token_sum,
                sqnr_overall=overall_sqnr,
                sqnr_per_layer=layer_sqnr_mean,
            )
    results[("fp32", "na")] = dict(loss=fp_loss, sqnr_overall=math.inf,
                                   sqnr_per_layer={})
    return results
