import torch
from scipy.linalg import hadamard
import math
import torch.nn.functional as F
from hadamard_transform import hadamard_transform

def outside_hadamard_transform(x, scale=1.0):
    return hadamard_transform(x) * scale


def fast_hadamard_transform(x, scale=1.0):
    """
    Fast implementation of Hadamard transform using recursive approach.
    This is more efficient than the naive matrix multiplication method.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim) where dim is the dimension to transform
        scale (float, optional): Scaling factor to apply after transformation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Transformed tensor of the same shape as input
    """
    # Store original shape and dimension
    x_shape = x.shape
    dim = x.shape[-1]
    
    # Reshape to 2D for easier processing
    x = x.reshape(-1, dim)
    
    # Calculate the next power of 2 for padding
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    
    # Pad the input if dimension is not a power of 2
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    
    # Perform fast Hadamard transform
    n = dim_padded
    while n > 1:
        half = n // 2
        # Reshape for butterfly operations
        x = x.view(-1, half, 2)
        # Perform butterfly operations
        x = torch.cat([x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]], dim=1)
        n = half
    
    # Apply scaling factor
    x = x * scale
    
    # Remove padding and restore original shape
    return x[..., :dim].reshape(*x_shape)

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

if __name__ == "__main__":
    # Test 1: Basic test with power of 2 dimension
    print("\nTest 1: Basic test with power of 2 dimension")
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    out1_naive = naive_hadamard_transform_with_scale(x1)
    out1_fast = fast_hadamard_transform(x1)
    out1_outside = outside_hadamard_transform(x1)
    print("Input:", x1)
    print("Naive Output:", out1_naive)
    print("Fast Output:", out1_fast)
    print("Outside Output:", out1_outside)
    print("Are all outputs equal?", torch.allclose(out1_naive, out1_fast) and torch.allclose(out1_naive, out1_outside))
    print("Shape:", out1_fast.shape)

    # Test 2: Test with non-power of 2 dimension
    print("\nTest 2: Test with non-power of 2 dimension")
    x2 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    out2_naive = naive_hadamard_transform_with_scale(x2)
    out2_fast = fast_hadamard_transform(x2)
    out2_outside = outside_hadamard_transform(x2)
    print("Input:", x2)
    print("Naive Output:", out2_naive)
    print("Fast Output:", out2_fast)
    print("Outside Output:", out2_outside)
    print("Are all outputs equal?", torch.allclose(out2_naive, out2_fast) and torch.allclose(out2_naive, out2_outside))
    print("Shape:", out2_fast.shape)

    # Test 3: Test with scaling factor
    print("\nTest 3: Test with scaling factor")
    x3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    out3_naive = naive_hadamard_transform_with_scale(x3, scale=0.5)
    out3_fast = fast_hadamard_transform(x3, scale=0.5)
    out3_outside = outside_hadamard_transform(x3, scale=0.5)
    print("Input:", x3)
    print("Naive Output (scaled by 0.5):", out3_naive)
    print("Fast Output (scaled by 0.5):", out3_fast)
    print("Outside Output (scaled by 0.5):", out3_outside)
    print("Are all outputs equal?", torch.allclose(out3_naive, out3_fast) and torch.allclose(out3_naive, out3_outside))

    # Test 4: Test with batch processing
    print("\nTest 4: Test with batch processing")
    x4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out4_naive = naive_hadamard_transform_with_scale(x4)
    out4_fast = fast_hadamard_transform(x4)
    out4_outside = outside_hadamard_transform(x4)
    print("Input:", x4)
    print("Naive Output:", out4_naive)
    print("Fast Output:", out4_fast)
    print("Outside Output:", out4_outside)
    print("Are all outputs equal?", torch.allclose(out4_naive, out4_fast) and torch.allclose(out4_naive, out4_outside))
    print("Shape:", out4_fast.shape)

    # Test 5: Test with larger dimension
    print("\nTest 5: Test with larger dimension")
    x5 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    out5_naive = naive_hadamard_transform_with_scale(x5)
    out5_fast = fast_hadamard_transform(x5)
    out5_outside = outside_hadamard_transform(x5)
    print("Input:", x5)
    print("Naive Output:", out5_naive)
    print("Fast Output:", out5_fast)
    print("Outside Output:", out5_outside)
    print("Are all outputs equal?", torch.allclose(out5_naive, out5_fast) and torch.allclose(out5_naive, out5_outside))
    print("Shape:", out5_fast.shape)

    # Performance comparison
    print("\nPerformance comparison:")
    import time
    
    # Create a larger tensor for timing
    x_large = torch.randn(1000, 1024, dtype=torch.float32)
    
    # Time naive implementation
    start_time = time.time()
    _ = naive_hadamard_transform_with_scale(x_large)
    naive_time = time.time() - start_time
    
    # Time fast implementation
    start_time = time.time()
    _ = fast_hadamard_transform(x_large)
    fast_time = time.time() - start_time
    
    # Time outside implementation
    start_time = time.time()
    _ = outside_hadamard_transform(x_large)
    outside_time = time.time() - start_time
    
    print(f"Naive implementation time: {naive_time:.4f} seconds")
    print(f"Fast implementation time: {fast_time:.4f} seconds")
    print(f"Outside implementation time: {outside_time:.4f} seconds")
    print(f"Speedup factor (naive/fast): {naive_time/fast_time:.2f}x")
    print(f"Speedup factor (naive/outside): {naive_time/outside_time:.2f}x")
    print(f"Speedup factor (fast/outside): {fast_time/outside_time:.2f}x")