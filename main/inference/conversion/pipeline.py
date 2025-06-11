import os
import sys
import torch
import faiss

import numpy as np
import torch.nn.functional as F

from scipy import signal

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.library.utils import get_providers
from main.library.predictors.Generator import Generator
from main.inference.conversion.utils import change_rms, clear_gpu_cache

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class Pipeline:
    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.is_half = config.is_half
        self.f0_generator = Generator(self.sample_rate, self.window, self.f0_min, self.f0_max, self.is_half, self.device, get_providers(), False)
    
    def extract_features(self, model, feats, version):
        return torch.as_tensor(model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {"feats": feats.detach().cpu().numpy()})[0 if version == "v1" else 1], dtype=torch.float32, device=feats.device)

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        pitch_guidance = pitch != None and pitchf != None
        feats = (torch.from_numpy(audio0).half() if self.is_half else torch.from_numpy(audio0).float())

        if feats.dim() == 2: feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        with torch.no_grad():
            if self.embed_suffix == ".pt":
                padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
                logits = model.extract_features(**{"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12})
                feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
            elif self.embed_suffix == ".onnx": feats = self.extract_features(model, feats.to(self.device), version).to(self.device)
            elif self.embed_suffix == ".safetensors":
                logits = model(feats.to(self.device))["last_hidden_state"]
                feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
            else: raise ValueError(translations["option_not_valid"])

            if protect < 0.5 and pitch_guidance: feats0 = feats.clone()

            if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
                npy = feats[0].cpu().numpy()
                if self.is_half: npy = npy.astype(np.float32)

                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)

                npy = np.sum(big_npy[ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
                if self.is_half: npy = npy.astype(np.float16)

                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if protect < 0.5 and pitch_guidance: feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            p_len = audio0.shape[0] // self.window

            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch_guidance: pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]

            if protect < 0.5 and pitch_guidance:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = ((net_g.infer(feats.half() if self.is_half else feats.float(), p_len, pitch if pitch_guidance else None, (pitchf.half() if self.is_half else pitchf.float()) if pitch_guidance else None, sid)[0][0, 0]).data.cpu().float().numpy()) if self.suffix == ".pth" else (net_g.run([net_g.get_outputs()[0].name], ({net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32), net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64), net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32)} if pitch_guidance else {net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32)}))[0][0, 0])

        if self.embed_suffix == ".pt": del padding_mask
        del feats, p_len, net_g
        clear_gpu_cache()

        return audio1
    
    def pipeline(self, logger, model, net_g, sid, audio, f0_up_key, f0_method, file_index, index_rate, pitch_guidance, filter_radius, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength, suffix, embed_suffix, f0_file=None, f0_onnx=False, pbar=None, proposal_pitch=False):
        self.suffix = suffix
        self.embed_suffix = embed_suffix

        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        pbar.update(1)
        opt_ts, audio_opt = [], []
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        t, inp_f0 = None, None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    raw_lines = f.read()

                    if len(raw_lines) > 0:
                        inp_f0 = []

                        for line in raw_lines.strip("\n").split("\n"):
                            inp_f0.append([float(i) for i in line.split(",")])

                        inp_f0 = np.array(inp_f0, dtype=np.float32)
            except:
                logger.error(translations["error_readfile"])
                inp_f0 = None

        pbar.update(1)
        if pitch_guidance:
            self.f0_generator.hop_length, self.f0_generator.f0_onnx_mode = hop_length, f0_onnx
            pitch, pitchf = self.f0_generator.calculator(self.x_pad, f0_method, audio_pad, f0_up_key, p_len, filter_radius, f0_autotune, f0_autotune_strength, manual_f0=inp_f0, proposal_pitch=proposal_pitch)

            if self.device == "mps": pitchf = pitchf.astype(np.float32)
            pitch, pitchf = torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()

        pbar.update(1)
        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])    
            s = t
            
        audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope)
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max

        if pitch_guidance: del pitch, pitchf
        del sid

        clear_gpu_cache()
        pbar.update(1)
        return audio_opt