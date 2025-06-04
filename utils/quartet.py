import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base_linear import QUANTIZER_CLASSES, NoQuantizer
from .quartet_quantizers import StochasticRoundingQuantizer

class QuartetBackwardFn(Function):
    @staticmethod
    def forward(ctx, x, w, g_quantizer):
        ctx.save_for_backward(x, w)
        ctx.g_quantizer = g_quantizer
        return F.linear(x, w, None)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        gq = ctx.g_quantizer
        if hasattr(gq, 're_randomize'):
            gq.re_randomize()
        grad_x = F.linear(gq(grad_output), gq(w.t().contiguous()), None)
        batch_seq_dim = math.prod(x.shape[:-1])
        grad_w = torch.einsum(
            'ib,jb->ij',
            gq(grad_output.reshape(batch_seq_dim, -1).t().contiguous()),
            gq(x.reshape(batch_seq_dim, -1).t().contiguous()),
        )
        return grad_x, grad_w, None

class QuartetScheme(nn.Module):
    def __init__(self, g_quantizer=None):
        super().__init__()
        if g_quantizer is None:
            g_quantizer = StochasticRoundingQuantizer()
        self.g_quantizer = g_quantizer

    def forward(self, x, w):
        return QuartetBackwardFn.apply(x, w, self.g_quantizer)

class QuartetLinear(nn.Linear):
    def __init__(self, in_features, out_features, *, weight_quantizer=None, activation_quantizer=None, gradient_quantizer=None, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        if gradient_quantizer is None:
            gradient_quantizer = StochasticRoundingQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        self.backward_scheme = QuartetScheme(gradient_quantizer)

    def forward(self, x):
        xq = self.activation_quantizer(x)
        wq = self.weight_quantizer(self.weight)
        return self.backward_scheme(xq, wq)

def prepare_model_for_quartet_training(model, args, target_module):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_quartet_training(module, args, target_module)
        if isinstance(module, nn.Linear):
            if name not in target_module:
                continue
            bias_data = module.bias.data if module.bias is not None else None
            weight_data = module.weight.data
            bias = module.bias is not None
            new_layer = QuartetLinear(
                module.in_features,
                module.out_features,
                bias=bias,
                weight_quantizer=QUANTIZER_CLASSES[args.w_quant](**args.w_quant_kwargs),
                activation_quantizer=QUANTIZER_CLASSES[args.a_quant](**args.a_quant_kwargs),
                gradient_quantizer=StochasticRoundingQuantizer(bits=args.weight_bits)
            )
            new_layer.weight.data.copy_(weight_data)
            if bias_data is not None:
                new_layer.bias.data.copy_(bias_data)
            model._modules[name] = new_layer
    return model
