import os
import torch
import random
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def MSE_loss(pred_v, tar_v, mass=None, mass_weighted=True):
    if mass_weighted:
        assert mass is not None, "Mass must be provided for mass-weighted loss"
        # L = ∑_v m_v * ‖v_tar - v_pred‖^2 / ∑_v m_v / num_channels
        return (
            (mass.unsqueeze(-1) * (pred_v - tar_v).pow(2)).sum(dim=(1, 2)) /
            (mass.sum(dim=1) * pred_v.shape[-1])
        ).mean()
    else:
        return ((pred_v - tar_v).pow(2)).mean()

def tensor_stats(name, x):
    print(f"{name}: {x.shape} | {x.dtype} | {x.device} | min: {x.min().item():.5f} | max: {x.max().item():.5f} | mean: {x.mean().item():.5f} | std: {x.std().item():.5f}")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    raise ValueError('Called to_np on an unsupported type')

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    out = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()
    return out

def torch_sparse_block_diag(matrices):
    """
    Create a block diagonal matrix from a list of 2D sparse torch tensors.
    """
    rows, cols = [], []
    values = []
    row_offset, col_offset = 0, 0
    
    for mat in matrices:
        row, col = mat._indices()
        rows.append(row + row_offset)
        cols.append(col + col_offset)
        values.append(mat._values())
        
        row_offset += mat.shape[0]
        col_offset += mat.shape[1]
    
    rows = torch.cat(rows, dim=0)
    cols = torch.cat(cols, dim=0)
    values = torch.cat(values, dim=0)
    
    indices = torch.stack([rows, cols], dim=0)
    shape = (row_offset, col_offset)
    
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    # https://github.com/nmwsharp/diffusion-net
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss