import os
import gc
import sys
import torch
import librosa

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library import torch_amd

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def clear_gpu_cache():
    gc.collect()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif torch_amd.is_available(): torch_amd.pytorch_ocl.empty_cache()

def extract_median_f0(f0):
    f0 = np.where(f0 == 0, np.nan, f0)
    return float(np.median(np.interp(np.arange(len(f0)), np.where(~np.isnan(f0))[0], f0[~np.isnan(f0)])))

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    return max(-limit, min(limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0))))))