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
        if os.getenv('WEIGHT_FT', 'false').lower() == 'true':
            weight_fp4 = fake_quant_fp4(x=weight, 
                                        stochastic_rounding=False, 
                                        block_size=int(os.getenv('BLOCK_SIZE')), 
                                        scale_format=os.getenv('SCALE_FORMAT'))
            input_fp4 = fake_quant_fp4(x=input, 
                                       stochastic_rounding=False, 
                                       block_size=int(os.getenv('BLOCK_SIZE')), 
                                       scale_format=os.getenv('SCALE_FORMAT'))
            out = input_fp4 @ weight_fp4.T
        else:
            if os.getenv('INPUT_FP4', 'false').lower() == 'true':
                input_fp4 = fake_quant_fp4(x=input, 
                                        stochastic_rounding=False, 
                                        block_size=int(os.getenv('BLOCK_SIZE')), 
                                        scale_format=os.getenv('SCALE_FORMAT'))
            else:
                input_fp4 = input

            if (os.getenv('WEIGHT_GRID', 'false').lower() == 'false') and (os.getenv('WEIGHT_FP4', 'false').lower() == 'true'):
                weight_fp4 = fake_quant_fp4(x=weight, 
                                            stochastic_rounding=False, 
                                            block_size=int(os.getenv('BLOCK_SIZE')), 
                                            scale_format=os.getenv('SCALE_FORMAT'))
            else:
                weight_fp4 = weight


            out = input_fp4 @ weight_fp4.T

        ctx.save_for_backward(
            input,
            weight,
        )
        ctx.first_call = first_call

        return out
    

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        (
            input,
            weight
        ) = ctx.saved_tensors

        if os.getenv('WEIGHT_FT', 'false').lower() == 'true':
            grad_input = grad_output @ weight
            grad_weight = grad_output.T @ input
        else:
            if os.getenv('GRAD_FP4', 'false').lower() == 'true':
                grad_output_fp4 = fake_quant_fp4(x=grad_output, 
                                                stochastic_rounding=os.getenv('GRAD_SR', 'false').lower() == 'true', 
                                                block_size=int(os.getenv('BLOCK_SIZE')), 
                                                scale_format=os.getenv('SCALE_FORMAT'))
            else:
                grad_output_fp4 = grad_output

            if (os.getenv('WEIGHT_GRID', 'false').lower() == 'false') and (os.getenv('WEIGHT_FP4', 'false').lower() == 'true'):
                weight_fp4 = fake_quant_fp4(x=weight.T, 
                                            stochastic_rounding=False, 
                                            block_size=int(os.getenv('BLOCK_SIZE')), 
                                            scale_format=os.getenv('SCALE_FORMAT')).T
            else:
                weight_fp4 = weight
            
            grad_input = grad_output_fp4 @ weight_fp4

            
            if os.getenv('GRAD_FP4', 'false').lower() == 'true':
                grad_output_fp4_t = fake_quant_fp4(x=grad_output.T, 
                                                stochastic_rounding=os.getenv('GRAD_SR', 'false').lower() == 'true', 
                                                block_size=int(os.getenv('BLOCK_SIZE')), 
                                                scale_format=os.getenv('SCALE_FORMAT'))
            else:
                grad_output_fp4_t = grad_output.T
                
            if os.getenv('INPUT_FP4', 'false').lower() == 'true':
                input_fp4 = fake_quant_fp4(x=input.T, 
                                        stochastic_rounding=os.getenv('INPUT_SR', 'false').lower() == 'true', 
                                        block_size=int(os.getenv('BLOCK_SIZE')), 
                                        scale_format=os.getenv('SCALE_FORMAT')).T
            else:
                input_fp4 = input
            
            grad_weight = grad_output_fp4_t @ input_fp4

        return grad_input, grad_weight, None
    

## FP4 All the Way
class FP4ATWLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,) -> None:
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.first_call = True

    def forward(self,
                input:torch.Tensor) -> torch.Tensor:

        A, B, C = input.shape
        out = FP4LinearF.apply(input.reshape(A * B, C), self.weight, self.first_call)
        self.first_call = False
        return out.reshape(A, B, -1)
    

        

        

