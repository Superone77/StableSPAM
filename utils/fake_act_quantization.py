import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .dummy_quant import fake_tensor_quantize, fake_tensor_dequantize


class A8Linear(torch.autograd.Function):
    ### Implement the forward and backward pass for the A8Linear layer with block-wise ACTIVATION quantization
    # Define quantization parameters as class attributes
    quant_dim = -1
    quant_group_size = 64
    quant_bit = 8
    quant_topk = 0
    quant_type = 'linear'
    stochastic_round = False
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (BS, N, C_in) in fp16
        # weigth: (C_out, C_in) in fp16??
        # Quantize x with specified parameters
        qx = fake_tensor_quantize(
            x, 
            dim=A8Linear.quant_dim, 
            group_size=A8Linear.quant_group_size, 
            bit=A8Linear.quant_bit, 
            topk=A8Linear.quant_topk, 
            quant_type=A8Linear.quant_type,
            stochastic=A8Linear.stochastic_round
        )
        
        # Save tensors and parameters needed for the backward pass
        ctx.save_for_backward(qx, weight, bias)
        ctx.x_shape = x.shape  # Save original shape separately
        
        # print(f"I am checking x quantization -> {x.shape} -> {qx.shape}")

        # Perform dequantization for forward computation
        def forward_a_float_activation(weight, x_dtype):
            # Dequantize with stored parameters
            float_x = fake_tensor_dequantize(
                qx, 
                dim=A8Linear.quant_dim, 
                shape=ctx.x_shape, 
                group_size=A8Linear.quant_group_size, 
                bit=A8Linear.quant_bit, 
                dtype=x_dtype, 
                topk=A8Linear.quant_topk, 
                quant_type=A8Linear.quant_type
            )
            return float_x @ weight.t() + bias if bias is not None else float_x @ weight.t()

        # Determine x dtype
        if x.dtype == torch.half:
            x_dtype = 'fp16'
        elif x.dtype == torch.bfloat16:
            x_dtype = 'bf16'
        else:
            x_dtype = 'fp32'

        # Apply forward computation with quantized activation
        output = forward_a_float_activation(weight, x_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with dequantized activations.
        """
        # Retrieve saved tensors
        qx, weight, bias = ctx.saved_tensors
        x_shape = ctx.x_shape  # Retrieve original shape

        # Determine weight dtype for dequantization
        if weight.dtype == torch.half:
            x_dtype = 'fp16'
        elif weight.dtype == torch.bfloat16:
            x_dtype = 'bf16'
        else:
            x_dtype = 'fp32'

        # Dequantize activations for gradient computation
        x = fake_tensor_dequantize(
            qx, 
            dim=A8Linear.quant_dim, 
            shape=x_shape, 
            group_size=A8Linear.quant_group_size, 
            bit=A8Linear.quant_bit, 
            dtype=x_dtype, 
            topk=A8Linear.quant_topk, 
            quant_type=A8Linear.quant_type
        )

        # Compute gradients
        grad_input = grad_output @ weight
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None
        
        # Compute gradient with respect to weight
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ x.reshape(-1, x.shape[-1])

        return grad_input, grad_weight, grad_bias

        # TODO(hanqing): not sure do we need this?
        # out_features, in_features = weight.shape
        # gradient accumulation
        # NOTE(hanqing): I don't think we need this, since we are not using the float_grad in the backward pass
        # if not hasattr(weight, 'float_grad'):
        #     weight.__setattr__('float_grad', None)

        # if weight.float_grad is not None:
        #     weight.float_grad += grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features) 
        # else:
        #     weight.float_grad = grad_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)

        # return grad_input, None, grad_bias

class QLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device='cpu', dtype=None, weight_data=None, bias_data=None, num_bits=8, group_size=64, stochastic_round=True, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight.__setattr__('stochastic_round', stochastic_round)
        self.weight.__setattr__('group_size', group_size)

        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias is not None:
            self.bias.data.copy_(bias_data)

        # Set quantization parameters for the custom layer
        A8Linear.quant_bit = num_bits
        A8Linear.quant_group_size = group_size
        A8Linear.stochastic_round = stochastic_round
        A8Linear.quant_dim = -1
        A8Linear.quant_topk = 0

    def forward(self, input: Tensor) -> Tensor:
        output = A8Linear.apply(input, self.weight, self.bias)
        return output
    
    def change_quant_bit(num_bits):
        A8Linear.quant_bit = num_bits
    
    def change_quant_group_size(group_size):
        A8Linear.quant_group_size = group_size

def prepare_model_for_act_int8_training_simulation(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_act_int8_training_simulation(module, args, target_module)

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
            new_layers = QLinear(in_features, out_features, bias=bias, device='cuda:0', 
                weight_data=weight_data, bias_data=bias_data, 
                num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)

            model._modules[name] = new_layers

    return model

if __name__ == '__main__':
    GROUP_SIZE=32
    fp16_linear1 = nn.Linear(4096, 4096, bias=False).to('cuda:0').to(torch.bfloat16)
    print('after initial weight for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    mem_weight_float = torch.cuda.memory_allocated('cuda:0')//1024/1024
    x = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:0', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    start = time.time()
    output = fp16_linear1(x)
    print('after forward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    output.sum().backward()
    print('output_full', output)
    end = time.time()
    print('after backward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('Gradient for weight:', fp16_linear1.weight.grad)
    print('------------------------------------')

    int8_simluate_linear1 = QLinear(4096, 4096, device='cuda:2', bias=False, num_bits=8, group_size=GROUP_SIZE, weight_data=fp16_linear1.weight.data, bias_data=None).to(torch.bfloat16).to('cuda:2')
    
    print('after initial weight for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:2')//1024/1024
    x2 = x.to('cuda:2')
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    start = time.time()
    output_int8_simulate = int8_simluate_linear1(x2)
    print('after forward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    output_int8_simulate.sum().backward()
    print('output_quant_simulation', output_int8_simulate)
    end = time.time()
    print('after backward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('Gradient for weight:', int8_simluate_linear1.weight.grad)
    print('------------------------------------')












