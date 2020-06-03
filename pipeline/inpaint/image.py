import torch

MEAN = [0.482, 0.447, 0.467]
STD = [0.230, 0.227, 0.232]

LAMBDA_DICT = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0
}

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


def normalize(x):
    x = x.transpose(1, 3)
    x = (x - torch.Tensor(MEAN)) / torch.Tensor(STD)
    x = x.transpose(1, 3)
    return x
