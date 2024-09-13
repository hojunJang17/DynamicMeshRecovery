import torch

def generate_padding_mask1d(nblock, nframes):
    padding_mask_1d = torch.zeros(nframes).bool()
    padding_mask_1d[:nblock] = True
    padding_mask_1d = padding_mask_1d[torch.randperm(nframes)]
    
    return padding_mask_1d