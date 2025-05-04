import math, torch
LOG_EPS = 1e-12

def calc_sqnr(x, x_hat):
    mse = (x - x_hat).pow(2).mean()
    power = x.pow(2).mean()
    return 10 * torch.log10(power / (mse + LOG_EPS))
