import os
import torch

from torch import nn
from io import BytesIO
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_model(configs, input_path):
    with open(input_path, "rb") as f:
        data = f.read()

    with open(os.path.join(configs["binary_path"], "decrypt.bin"), "rb") as f:
        key = f.read()

    return BytesIO(unpad(AES.new(key, AES.MODE_CBC, data[:16]).decrypt(data[16:]), AES.block_size)).read()

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d: l2_loss.append((module.weight**2).sum() / 2.0)

    return l2_alpha * sum(l2_loss)

def torch_interp(x, xp, fp):
    sort_idx = torch.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[sort_idx]

    right_idxs = torch.searchsorted(xp, x).clamp(max=len(xp) - 1)
    left_idxs = (right_idxs - 1).clamp(min=0)
    x_left = xp[left_idxs]
    y_left = fp[left_idxs]

    interp_vals = y_left + ((x - x_left) * (fp[right_idxs] - y_left) / (xp[right_idxs] - x_left))
    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]

    return interp_vals

def batch_interp_with_replacement_detach(uv, f0):
    result = f0.clone()
    for i in range(uv.shape[0]):
        interp_vals = torch_interp(torch.where(uv[i])[-1], torch.where(~uv[i])[-1], f0[i][~uv[i]]).detach()
        result[i][uv[i]] = interp_vals
        
    return result

def ensemble_f0(f0s, key_shift_list, tta_uv_penalty):
    device = f0s.device
    f0s = f0s / (torch.pow(2, torch.tensor(key_shift_list, device=device).to(device).unsqueeze(0).unsqueeze(0) / 12))
    notes = torch.log2(f0s / 440) * 12 + 69
    notes[notes < 0] = 0

    uv_penalty = tta_uv_penalty**2
    dp = torch.zeros_like(notes, device=device)
    backtrack = torch.zeros_like(notes, device=device).long()
    dp[:, 0, :] = (notes[:, 0, :] <= 0) * uv_penalty

    for t in range(1, notes.size(1)):
        penalty = torch.zeros([notes.size(0), notes.size(2), notes.size(2)], device=device)
        t_uv = notes[:, t, :] <= 0
        penalty += uv_penalty * t_uv.unsqueeze(1)

        t1_uv = notes[:, t - 1, :] <= 0
        l2 = torch.pow((notes[:, t - 1, :].unsqueeze(-1) - notes[:, t, :].unsqueeze(1)) * (~t1_uv).unsqueeze(-1) * (~t_uv).unsqueeze(1), 2) - 0.5
        l2 = l2 * (l2 > 0)

        penalty += l2
        penalty += t1_uv.unsqueeze(-1) * (~t_uv).unsqueeze(1) * uv_penalty * 2

        min_value, min_indices = torch.min(dp[:, t - 1, :].unsqueeze(-1) + penalty, dim=1)
        dp[:, t, :] = min_value
        backtrack[:, t, :] = min_indices

    t = f0s.size(1) - 1
    f0_result = torch.zeros_like(f0s[:, :, 0], device=device)
    min_indices = torch.argmin(dp[:, t, :], dim=-1)

    for i in range(0, t + 1):
        f0_result[:, t - i] = f0s[:, t - i, min_indices]
        min_indices = backtrack[:, t - i, min_indices]

    return f0_result.unsqueeze(-1)

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims == 2"
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()