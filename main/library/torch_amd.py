import torch
import platform
import subprocess

import numpy as np
import torch.nn.functional as F

from librosa.util import pad_center
from scipy.signal import get_window

try:
    import pytorch_ocl
except:
    pytorch_ocl = None

torch_available = pytorch_ocl != None

def get_amd_gpu_windows():
    try:
        return [gpu.strip() for gpu in subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode().split('\n')[1:] if 'AMD' in gpu or 'Radeon' in gpu or 'Vega' in gpu]
    except:
        return []

def get_amd_gpu_linux():
    try:
        return [gpu for gpu in subprocess.check_output("lspci | grep VGA", shell=True).decode().split('\n') if 'AMD' in gpu or 'Radeon' in gpu or 'Vega' in gpu]
    except:
        return []

def get_gpu_list():
    return (get_amd_gpu_windows() if platform.system() == "Windows" else get_amd_gpu_linux()) if torch_available else []

def device_count():
    return len(get_gpu_list()) if torch_available else 0

def device_name(device_id = 0):
    return (get_gpu_list()[device_id] if device_id >= 0 and device_id < device_count() else "") if torch_available else ""

def is_available():
    return (device_count() > 0) if torch_available else False

class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.pad_amount = int(self.filter_length / 2)
        self.win_length = win_length
        self.hann_window = {}

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        if win_length is None or not win_length: win_length = filter_length
        assert filter_length >= win_length

        fft_window = torch.from_numpy(pad_center(get_window(window, win_length, fftbins=True), size=filter_length)).float()
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, eps):
        input_data = F.pad(input_data, (self.pad_amount, self.pad_amount), mode="reflect")
        forward_transform = torch.matmul(self.forward_basis, input_data.unfold(1, self.filter_length, self.hop_length).permute(0, 2, 1))
        cutoff = int(self.filter_length / 2 + 1)

        return torch.sqrt(forward_transform[:, :cutoff, :]**2 + forward_transform[:, cutoff:, :]**2 + eps)

if torch_available:
    def _group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
        N, C = x.shape[:2]
        assert C % num_groups == 0

        shape = (N, num_groups, C // num_groups) + x.shape[2:]
        x_reshaped = x.view(shape)

        dims = (2,) + tuple(range(3, x_reshaped.dim()))
        mean = x_reshaped.mean(dim=dims, keepdim=True)
        var = x_reshaped.var(dim=dims, keepdim=True, unbiased=False)

        x_norm = (x_reshaped - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view_as(x)

        if weight is not None:
            weight = weight.view(1, C, *([1] * (x.dim() - 2)))
            x_norm = x_norm * weight

        if bias is not None:
            bias = bias.view(1, C, *([1] * (x.dim() - 2)))
            x_norm = x_norm + bias

        return x_norm
    
    def _script(f, *_, **__):
        f.graph = pytorch_ocl.torch._C.Graph()
        return f

    F.group_norm = _group_norm
    torch.jit.script = _script