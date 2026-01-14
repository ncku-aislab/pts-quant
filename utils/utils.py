import torch

def check_power_of_two(tensor):
    log2_values = torch.log2(tensor)
    return torch.all(torch.abs(log2_values.round() - log2_values) < 1e-6)