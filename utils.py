import numpy as np
import torch

def set_torch_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def nancorr(a,b):
    mask = np.isfinite(a)&np.isfinite(b)
    a,b = a[mask],b[mask]
    return np.corrcoef(a,b)[0,1]

def calc_distribution_overlap(dist1,dist2):
    bin_edges = np.histogram_bin_edges(np.concatenate([dist1, dist2]))
    hist1, _ = np.histogram(dist1, bins=bin_edges)
    hist2, _ = np.histogram(dist2, bins=bin_edges)
    return np.minimum(hist1, hist2).sum()/len(dist1)