import torch

def torch_cov(x1, x2):
    # input_vec: N, C
    N = x1.size(0)
    x1 = x1 - torch.mean(x1, axis=0, keepdim=True)
    x2 = x2 - torch.mean(x2, axis=0, keepdim=True)
    cov_matrix = torch.matmul(x1.T, x2) / (N - 1)
    return cov_matrix

def torch_corroef(x1, x2):
    # input_vec: N, C
    cov_matrix = torch_cov(x1, x2)
    x1_std = torch.std(x1, dim=0, keepdim=True)
    x2_std = torch.std(x2, dim=0, keepdim=True)
    std_matrix = torch.matmul(x1_std.T, x2_std)
    cov_matrix = cov_matrix / std_matrix.clamp(min=1e-4)
    return cov_matrix