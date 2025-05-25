import torch
from hadamard_transform import hadamard_transform

def naive_hadamard_transform_with_scale(x, scale=1.0):
    return hadamard_transform(x) * scale 