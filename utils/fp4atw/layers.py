import os
import torch
import torch.nn as nn

from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Function
from .fp4 import fake_quant_fp4

class FP4LinearF(Function):
    @staticmethod
    def forward(
        ctx,
        input:torch.Tensor,
        weight:torch.Tensor,
        first_call:bool,
    ) -> torch.Tensor:
        weight_fp4 = fake_quant_fp4(x=weight, 
                                    stochastic_rounding=False, 
                                    block_size=int(os.getenv('BLOCK_SIZE')), 
                                    scale_format=os.getenv('SCALE_FORMAT'))
        input_fp4 = fake_quant_fp4(x=input, 
                                    stochastic_rounding=False, 
                                    block_size=int(os.getenv('BLOCK_SIZE')), 
                                    scale_format=os.getenv('SCALE_FORMAT'))
        out = input_fp4 @ weight_fp4.T
        ctx.save_for_backward(
            input,
            weight,
        )
        ctx.first_call = first_call
        # print(torch.norm(weight).item(),
        #     torch.norm(weight_fp4).item(),
        #     torch.norm(input).item(),
        #     torch.norm(input_fp4).item())
        return out
    

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            input,
            weight
        ) = ctx.saved_tensors
        grad_output_fp4 = fake_quant_fp4(x=grad_output, 
                                        stochastic_rounding=True, 
                                        block_size=int(os.getenv('BLOCK_SIZE')), 
                                        scale_format=os.getenv('SCALE_FORMAT'))
        

        weight_fp4 = fake_quant_fp4(x=weight.T, 
                                    stochastic_rounding=False, 
                                    block_size=int(os.getenv('BLOCK_SIZE')), 
                                    scale_format=os.getenv('SCALE_FORMAT')).T
        
        grad_input = grad_output_fp4 @ weight_fp4
        
        grad_output_fp4_t = fake_quant_fp4(x=grad_output.T, 
                                            stochastic_rounding=True, 
                                            block_size=int(os.getenv('BLOCK_SIZE')), 
                                            scale_format=os.getenv('SCALE_FORMAT'))

        input_fp4 = fake_quant_fp4(x=input.T, 
                                stochastic_rounding=os.getenv('INPUT_SR', 'false').lower() == 'true', 
                                block_size=int(os.getenv('BLOCK_SIZE')), 
                                scale_format=os.getenv('SCALE_FORMAT')).T
        
        
        grad_weight = grad_output_fp4_t @ input_fp4
        # print(torch.norm(grad_output).item(),
        #     torch.norm(grad_output_fp4).item(),
        #     torch.norm(weight).item(),
        #     torch.norm(weight_fp4).item(),
        #     torch.norm(input).item(),
        #     torch.norm(input_fp4).item(),)
        return grad_input, grad_weight, None
    

## FP4 All the Way
class FP4ATWLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,) -> None:
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.reset_parameters()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.first_call = True

    def forward(self,
                input:torch.Tensor) -> torch.Tensor:
        A, B, C = input.shape
        out = FP4LinearF.apply(input.reshape(A * B, C), self.weight, self.first_call)
        self.first_call = False
        return out.reshape(A, B, -1)
    

        

        

