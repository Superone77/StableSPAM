import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import pdb
from .act_weight_quantization import calc_sqnr
# ----------------------
# 1) 量化函数
# ----------------------

def fp4_fake_quantize(
    x: torch.Tensor,
    e: int = 1, 
    m: int = 2,
    b: int = 1,
    topk: int = 2  # Top-K 绝对值最大元素单独处理的数量
): 
    x = x.clone()
    x_shape = x.shape
    x = x.view(-1,64) # group_size = 256
    if topk > 0:
        xabs = x.abs()      # 存储x中元素绝对值
        xabs_topk_index = xabs.topk(topk, dim=-1).indices #找到topk元素的索引
        topk_values = torch.gather(x, 1 , xabs_topk_index)#把topk元素提取出来
        x[torch.arange(0, x.size(0), device=x.device)[:, None].expand(-1, topk), xabs_topk_index] = 0 # 把topk元素置为0
    alpha = x.abs().max(dim=-1).values.clamp(min=1e-6) #############SCALE FACTOR？？？###################
    q_max = alpha                                           #############SCALE FACTOR？？？###################
    q_min = -q_max                                          #############SCALE FACTOR？？？###################
    x_clamped = torch.clamp(x, q_min[:, None], q_max[:, None])
    alpha_hat = alpha * (2**(-b))
    b_hat = 2**e - torch.log2(q_max) + torch.log2(torch.tensor(2 - 2**(-m), dtype=torch.float32)) - 1
    log_v = torch.floor(torch.log2(torch.abs(x_clamped) + 1e-8) + b_hat.unsqueeze(1))
    v = torch.pow(2, torch.clamp(log_v - m, min=1-m))
    if topk > 0:
        x = alpha_hat.unsqueeze(1) * v * torch.round(x_clamped / (alpha_hat.unsqueeze(1) * v)+1e-12 )
        row_indices = torch.arange(0, x.size(0), device=x.device).view(-1, 1).expand_as(xabs_topk_index)  # [1024, 2]
        x[row_indices, xabs_topk_index] = topk_values
    
    x = x.view(*x_shape)
    return x


import torch



def fp4_tensor_quantize(x: torch.Tensor):

    return fp4_fake_quantize(x)


# ----------------------
# 2) 自定义 Linear 前后向
# ----------------------
class A8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor):
        # ---- debug
        quant_x =fp4_tensor_quantize(x)
        sqnr = calc_sqnr(x, quant_x)
        ctx.save_for_backward(quant_x,weight,bias)
        output = quant_x @ weight.t()
        if bias is not None:
            output += bias
        return output, sqnr

    @staticmethod
    def backward(ctx, grad_output,sqnr):
        quant_x, weight, bias = ctx.saved_tensors

        grad_input =  grad_output @ weight

        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                                      quant_x.reshape(-1, quant_x.shape[-1])
        
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias


# ----------------------
# 3) 带权重量化的 QLinear
# ----------------------
class Qfp4Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_data=None,
        bias_data=None,
        num_bits=4,
        group_size=256,
        stochastic_round=True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.stochastic_round = stochastic_round
        self.sqnr_weight = 0
        self.sqnr_act = 0
        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias:
            self.bias.data.copy_(bias_data)
            

    def forward(self, input: Tensor) -> Tensor:
        # pdb.set_trace()
        qweight = fp4_tensor_quantize(self.weight)
        quant_w = qweight.detach() + self.weight - self.weight.detach()
        self.sqnr_weight = calc_sqnr(self.weight, qweight)
        # --- debug --- pdb.settrace
        # pdb.set_trace()
        output,self.sqnr_act = A8Linear.apply(input, quant_w, self.bias)
        return output
    

def prepare_model_for_fp4_training_simulation_act_weight(model, args, target_module):

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_fp4_training_simulation_act_weight(module, args, target_module)

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
            new_layers = Qfp4Linear(in_features, out_features, bias=bias, device='cuda:0', 
                weight_data=weight_data, bias_data=bias_data, 
                num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)

            model._modules[name] = new_layers
    return model



# ----------------------
# 4) 一个简单的测试
# ----------------------
if __name__ == "__main__":
    # 测试一下前向后向是否能跑通
    model = QLinear(4, 3, bias=True, device='cpu', dtype=torch.float32)
    x = torch.randn(2, 4, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    print("Input x:", x)
    print("Output y:", y)
    print("Weight grad:", model.weight.grad)
    print("Bias grad:", model.bias.grad)
