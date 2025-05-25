import torch
from scipy.linalg import hadamard
import math
import torch.nn.functional as F

def naive_hadamard_transform_with_scale(x, scale=1.0):
    """
    Performs a Hadamard transform on the input tensor with optional scaling.
    
    The Hadamard transform is a linear, orthogonal transform that decomposes a signal into 
    a set of Walsh functions. It's particularly useful in signal processing and data compression.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim) where dim is the dimension to transform
        scale (float, optional): Scaling factor to apply after transformation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Transformed tensor of the same shape as input
        
    Note:
        - The input dimension must be padded to the next power of 2 if it's not already
        - The transform is applied along the last dimension
        - The output is scaled by the provided scale factor
    """
    if hadamard is None:
        raise ImportError("Please install scipy")

    # Store original shape for later reshaping
    x_shape = x.shape
    dim = x.shape[-1]
    
    # Reshape to 2D for easier processing: (batch_size, dim)
    x = x.reshape(-1, dim)
    
    # Calculate the next power of 2 for padding
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    
    # Pad the input if dimension is not a power of 2
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    
    # Create Hadamard matrix and perform the transform
    # The Hadamard matrix is a square matrix of size 2^n with entries ±1
    # The transform is equivalent to matrix multiplication with this matrix
    hadamard_matrix = torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device)
    out = F.linear(x, hadamard_matrix)
    
    # Apply scaling factor
    out = out * scale
    
    # Remove padding and restore original shape
    return out[..., :dim].reshape(*x_shape)