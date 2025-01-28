import re
import os
import sys
import time
import faiss
import torch
import shutil
import librosa
import logging
import argparse
import warnings
import parselmouth
import onnxruntime
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from tqdm import tqdm
from scipy import signal
from distutils.util import strtobool
from fairseq import checkpoint_utils

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.predictors.FCPE import FCPE
from main.library.predictors.RMVPE import RMVPE
from main.library.predictors.WORLD import PYWORLD
from main.library.algorithm.synthesizers import Synthesizer
from main.library.predictors.CREPE import predict, mean, median
from main.library.utils import check_predictors, check_embedders, load_audio, process_audio, merge_audio

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
config = Config()
translations = config.translations
logger = logging.getLogger(__name__)
logger.propagate = False

for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "numba.core", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join("assets", "logs", "convert.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--volume_envelope", type=float, default=1)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/output.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--pth_path",  type=str,  required=True)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--split_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def main():
    args = parse_arguments()
    pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, checkpointing = args.pitch, args.filter_radius, args.index_rate, args.volume_envelope,args.protect, args.hop_length, args.f0_method, args.input_path, args.output_path, args.pth_path, args.index_path, args.f0_autotune, args.f0_autotune_strength, args.clean_audio, args.clean_strength, args.export_format, args.embedder_model, args.resample_sr, args.split_audio, args.checkpointing

    log_data = {translations['pitch']: pitch, translations['filter_radius']: filter_radius, translations['index_strength']: index_rate, translations['volume_envelope']: volume_envelope, translations['protect']: protect, "Hop length": hop_length, translations['f0_method']: f0_method, translations['audio_path']: input_path, translations['output_path']: output_path.replace('wav', export_format), translations['model_path']: pth_path, translations['indexpath']: index_path, translations['autotune']: f0_autotune, translations['clear_audio']: clean_audio, translations['export_format']: export_format, translations['hubert_model']: embedder_model, translations['split_audio']: split_audio, translations['memory_efficient_training']: checkpointing}

    if clean_audio: log_data[translations['clean_strength']] = clean_strength
    if resample_sr != 0: log_data[translations['sample_rate']] = resample_sr
    if f0_autotune: log_data[translations['autotune_rate_info']] = f0_autotune_strength

    logger.debug("\n\n".join([f"{key}: {value}" for key, value in log_data.items()]))

    check_predictors(f0_method)
    check_embedders(embedder_model)
    
    run_convert_script(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, input_path=input_path, output_path=output_path, pth_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, split_audio=split_audio, checkpointing=checkpointing)

def run_batch_convert(params):
    path, audio_temp, export_format, cut_files, pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, embedder_model, resample_sr, checkpointing = params["path"], params["audio_temp"], params["export_format"], params["cut_files"], params["pitch"], params["filter_radius"], params["index_rate"], params["volume_envelope"], params["protect"], params["hop_length"], params["f0_method"], params["pth_path"], params["index_path"], params["f0_autotune"], params["f0_autotune_strength"], params["clean_audio"], params["clean_strength"], params["embedder_model"], params["resample_sr"], params["checkpointing"]

    segment_output_path = os.path.join(audio_temp, f"output_{cut_files.index(path)}.{export_format}")
    if os.path.exists(segment_output_path): os.remove(segment_output_path)
    
    VoiceConverter().convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=path, audio_output_path=segment_output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing)
    os.remove(path)

    if os.path.exists(segment_output_path): return segment_output_path
    else: 
        logger.warning(f"{translations['not_found_convert_file']}: {segment_output_path}")
        sys.exit(1)

def run_convert_script(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, checkpointing):
    cvt = VoiceConverter()
    start_time = time.time()

    pid_path = os.path.join("assets", "convert_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
        logger.warning(translations["provide_file"].format(filename=translations["model"]))
        sys.exit(1)

    output_dir = os.path.dirname(output_path) or output_path
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    processed_segments = []
    audio_temp = os.path.join("audios_temp")
    if not os.path.exists(audio_temp) and split_audio: os.makedirs(audio_temp, exist_ok=True)

    if os.path.isdir(input_path):
        try:
            logger.info(translations["convert_batch"])
            audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

            if not audio_files: 
                logger.warning(translations["not_found_audio"])
                sys.exit(1)

            logger.info(translations["found_audio"].format(audio_files=len(audio_files)))

            for audio in audio_files:
                audio_path = os.path.join(input_path, audio)
                output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

                if split_audio:
                    try:
                        cut_files, time_stamps = process_audio(logger, audio_path, audio_temp)
                        params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr, "checkpointing": checkpointing} for path in cut_files]
                        
                        with tqdm(total=len(params_list), desc=translations["convert_audio"], ncols=100, unit="a") as pbar:
                            for params in params_list:
                                results = run_batch_convert(params)
                                processed_segments.append(results)

                                pbar.update(1)
                                logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))

                        merge_audio(processed_segments, time_stamps, audio_path, output_audio, export_format)
                    except Exception as e:
                        logger.error(translations["error_convert_batch"].format(e=e))
                    finally:
                        if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
                else:
                    try:
                        logger.info(f"{translations['convert_audio']} '{audio_path}'...")
                        if os.path.exists(output_audio): os.remove(output_audio)

                        with tqdm(total=1, desc=translations["convert_audio"], ncols=100, unit="a") as pbar:
                            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing)
                            pbar.update(1)
                            logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
                    except Exception as e:
                        logger.error(translations["error_convert"].format(e=e))

            elapsed_time = time.time() - start_time
            logger.info(translations["convert_batch_success"].format(elapsed_time=f"{elapsed_time:.2f}", output_path=output_path.replace('wav', export_format)))
        except Exception as e:
            logger.error(translations["error_convert_batch_2"].format(e=e))
    else:
        logger.info(f"{translations['convert_audio']} '{input_path}'...")

        if not os.path.exists(input_path):
            logger.warning(translations["not_found_audio"])
            sys.exit(1)
        
        if os.path.isdir(output_path): output_path = os.path.join(output_path, f"output.{export_format}")
        if os.path.exists(output_path): os.remove(output_path)

        if split_audio:
            try:              
                cut_files, time_stamps = process_audio(logger, input_path, audio_temp)
                params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr, "checkpointing": checkpointing} for path in cut_files]
                
                with tqdm(total=len(params_list), desc=translations["convert_audio"], ncols=100, unit="a") as pbar:
                    for params in params_list:
                        results = run_batch_convert(params)
                        processed_segments.append(results)

                        pbar.update(1)
                        logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))

                merge_audio(processed_segments, time_stamps, input_path, output_path.replace("wav", export_format), export_format)
            except Exception as e:
                logger.error(translations["error_convert_batch"].format(e=e))
            finally:
                if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
        else:
            try:
                with tqdm(total=1, desc=translations["convert_audio"], ncols=100, unit="a") as pbar:
                    cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing)
                    pbar.update(1)

                    logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
            except Exception as e:
                logger.error(translations["error_convert"].format(e=e))

        if os.path.exists(pid_path): os.remove(pid_path)
        elapsed_time = time.time() - start_time
        
        logger.info(translations["convert_audio_success"].format(input_path=input_path, elapsed_time=f"{elapsed_time:.2f}", output_path=output_path.replace('wav', export_format)))

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0

class VC:
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
        self.ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00,  207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict

    def get_providers(self):
        ort_providers = onnxruntime.get_available_providers()

        if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
        elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
        else: providers = ["CPUExecutionProvider"]

        return providers

    def get_f0_pm(self, x, p_len):
        f0 = (parselmouth.Sound(x, self.sample_rate).to_pitch_ac(time_step=self.window / self.sample_rate * 1000 / 1000, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])
        pad_size = (p_len - len(f0) + 1) // 2

        if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0
 
    def get_f0_mangio_crepe(self, x, p_len, hop_length, model="full", onnx=False):
        providers = self.get_providers() if onnx else None

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        audio = torch.unsqueeze(torch.from_numpy(x).to(self.device, copy=True), dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True).detach()

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(predict(audio.detach(), self.sample_rate, hop_length, self.f0_min, self.f0_max, model, batch_size=hop_length * 2, device=self.device, pad=True, providers=providers, onnx=onnx).squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_crepe(self, x, model="full", onnx=False):
        providers = self.get_providers() if onnx else None
        
        f0, pd = predict(torch.tensor(np.copy(x))[None].float(), self.sample_rate, self.window, self.f0_min, self.f0_max, model, batch_size=512, device=self.device, return_periodicity=True, providers=providers, onnx=onnx)
        f0, pd = mean(f0, 3), median(pd, 3)
        f0[pd < 0.1] = 0

        return f0[0].cpu().numpy()

    def get_f0_fcpe(self, x, p_len, hop_length, onnx=False, legacy=False):
        providers = self.get_providers() if onnx else None

        model_fcpe = FCPE(os.path.join("assets", "models", "predictors", "fcpe" + (".onnx" if onnx else ".pt")), hop_length=int(hop_length), f0_min=int(self.f0_min), f0_max=int(self.f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03, providers=providers, onnx=onnx) if legacy else FCPE(os.path.join("assets", "models", "predictors", "fcpe" + (".onnx" if onnx else ".pt")), hop_length=self.window, f0_min=0, f0_max=8000, dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.006, providers=providers, onnx=onnx)
        f0 = model_fcpe.compute_f0(x, p_len=p_len)

        del model_fcpe
        return f0
    
    def get_f0_rmvpe(self, x, legacy=False, onnx=False):
        providers = self.get_providers() if onnx else None

        rmvpe_model = RMVPE(os.path.join("assets", "models", "predictors", "rmvpe" + (".onnx" if onnx else ".pt")), device=self.device, onnx=onnx, providers=providers)
        f0 = rmvpe_model.infer_from_audio_with_pitch(x, thred=0.03, f0_min=self.f0_min, f0_max=self.f0_max) if legacy else rmvpe_model.infer_from_audio(x, thred=0.03)

        del rmvpe_model
        return f0

    def get_f0_pyworld(self, x, filter_radius, model="harvest"):
        if model == "harvest": f0, t = PYWORLD.harvest(x.astype(np.double),  fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        elif model == "dio": f0, t = PYWORLD.dio(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        else: raise ValueError(translations["method_not_valid"])

        f0 = PYWORLD.stonemask(x.astype(np.double), self.sample_rate, t, f0)

        if filter_radius > 2 or model == "dio": f0 = signal.medfilt(f0, 3)
        return f0
    
    def get_f0_yin(self, x, hop_length, p_len):
        source = np.array(librosa.yin(x.astype(np.double), sr=self.sample_rate, fmin=self.f0_min, fmax=self.f0_max, hop_length=hop_length))
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))
    
    def get_f0_pyin(self, x, hop_length, p_len):
        f0, _, _ = librosa.pyin(x.astype(np.double), fmin=self.f0_min, fmax=self.f0_max, sr=self.sample_rate, hop_length=hop_length)
        source = np.array(f0)
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_hybrid(self, methods_str, x, p_len, hop_length, filter_radius):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        f0_computation_stack, resampled_stack = [], []
        logger.debug(translations["hybrid_methods"].format(methods=methods))

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        for method in methods:
            f0 = None
            
            if method == "pm": f0 = self.get_f0_pm(x, p_len)
            elif method == "dio": f0 = self.get_f0_pyworld(x, filter_radius, "dio")
            elif method == "mangio-crepe-tiny": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny")
            elif method == "mangio-crepe-tiny-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=True)
            elif method == "mangio-crepe-small": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small")
            elif method == "mangio-crepe-small-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=True)
            elif method == "mangio-crepe-medium": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium")
            elif method == "mangio-crepe-medium-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=True)
            elif method == "mangio-crepe-large": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large")
            elif method == "mangio-crepe-large-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=True)
            elif method == "mangio-crepe-full": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full")
            elif method == "mangio-crepe-full-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=True)
            elif method == "crepe-tiny": f0 = self.get_f0_crepe(x, "tiny")
            elif method == "crepe-tiny-onnx": f0 = self.get_f0_crepe(x, "tiny", onnx=True)
            elif method == "crepe-small": f0 = self.get_f0_crepe(x, "small")
            elif method == "crepe-small-onnx": f0 = self.get_f0_crepe(x, "small", onnx=True)
            elif method == "crepe-medium": f0 = self.get_f0_crepe(x, "medium")
            elif method == "crepe-medium-onnx": f0 = self.get_f0_crepe(x, "medium", onnx=True)
            elif method == "crepe-large": f0 = self.get_f0_crepe(x, "large")
            elif method == "crepe-large-onnx": f0 = self.get_f0_crepe(x, "large", onnx=True)
            elif method == "crepe-full": f0 = self.get_f0_crepe(x, "full")
            elif method == "crepe-full-onnx": f0 = self.get_f0_crepe(x, "full", onnx=True)
            elif method == "fcpe": f0 = self.get_f0_fcpe(x, p_len, int(hop_length))
            elif method == "fcpe-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True)
            elif method == "fcpe-legacy": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True)
            elif method == "fcpe-legacy-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True, legacy=True)
            elif method == "rmvpe": f0 = self.get_f0_rmvpe(x)
            elif method == "rmvpe-onnx": f0 = self.get_f0_rmvpe(x, onnx=True)
            elif method == "rmvpe-legacy": f0 = self.get_f0_rmvpe(x, legacy=True)
            elif method == "rmvpe-legacy-onnx": f0 = self.get_f0_rmvpe(x, legacy=True, onnx=True)
            elif method == "harvest": f0 = self.get_f0_pyworld(x, filter_radius, "harvest") 
            elif method == "yin": f0 = self.get_f0_yin(x, int(hop_length), p_len)
            elif method == "pyin": f0 = self.get_f0_pyin(x, int(hop_length), p_len)
            else: raise ValueError(translations["method_not_valid"])
            
            f0_computation_stack.append(f0) 

        for f0 in f0_computation_stack:
            resampled_stack.append(np.interp(np.linspace(0, len(f0), p_len), np.arange(len(f0)), f0))

        return resampled_stack[0] if len(resampled_stack) == 1 else np.nanmedian(np.vstack(resampled_stack), axis=0)

    def get_f0(self, x, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength):
        if f0_method == "pm": f0 = self.get_f0_pm(x, p_len)
        elif f0_method == "dio": f0 = self.get_f0_pyworld(x, filter_radius, "dio")
        elif f0_method == "mangio-crepe-tiny": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny")
        elif f0_method == "mangio-crepe-tiny-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=True)
        elif f0_method == "mangio-crepe-small": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small")
        elif f0_method == "mangio-crepe-small-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=True)
        elif f0_method == "mangio-crepe-medium": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium")
        elif f0_method == "mangio-crepe-medium-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=True)
        elif f0_method == "mangio-crepe-large": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large")
        elif f0_method == "mangio-crepe-large-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=True)
        elif f0_method == "mangio-crepe-full": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full")
        elif f0_method == "mangio-crepe-full-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=True)
        elif f0_method == "crepe-tiny": f0 = self.get_f0_crepe(x, "tiny")
        elif f0_method == "crepe-tiny-onnx": f0 = self.get_f0_crepe(x, "tiny", onnx=True)
        elif f0_method == "crepe-small": f0 = self.get_f0_crepe(x, "small")
        elif f0_method == "crepe-small-onnx": f0 = self.get_f0_crepe(x, "small", onnx=True)
        elif f0_method == "crepe-medium": f0 = self.get_f0_crepe(x, "medium")
        elif f0_method == "crepe-medium-onnx": f0 = self.get_f0_crepe(x, "medium", onnx=True)
        elif f0_method == "crepe-large": f0 = self.get_f0_crepe(x, "large")
        elif f0_method == "crepe-large-onnx": f0 = self.get_f0_crepe(x, "large", onnx=True)
        elif f0_method == "crepe-full": f0 = self.get_f0_crepe(x, "full")
        elif f0_method == "crepe-full-onnx": f0 = self.get_f0_crepe(x, "full", onnx=True)
        elif f0_method == "fcpe": f0 = self.get_f0_fcpe(x, p_len, int(hop_length))
        elif f0_method == "fcpe-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True)
        elif f0_method == "fcpe-legacy": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True)
        elif f0_method == "fcpe-legacy-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True, legacy=True)
        elif f0_method == "rmvpe": f0 = self.get_f0_rmvpe(x)
        elif f0_method == "rmvpe-onnx": f0 = self.get_f0_rmvpe(x, onnx=True)
        elif f0_method == "rmvpe-legacy": f0 = self.get_f0_rmvpe(x, legacy=True)
        elif f0_method == "rmvpe-legacy-onnx": f0 = self.get_f0_rmvpe(x, legacy=True, onnx=True)
        elif f0_method == "harvest": f0 = self.get_f0_pyworld(x, filter_radius, "harvest") 
        elif f0_method == "yin": f0 = self.get_f0_yin(x, int(hop_length), p_len)
        elif f0_method == "pyin": f0 = self.get_f0_pyin(x, int(hop_length), p_len)
        elif "hybrid" in f0_method: f0 = self.get_f0_hybrid(f0_method, x, p_len, hop_length, filter_radius)
        else: raise ValueError(translations["method_not_valid"])

        if f0_autotune: f0 = Autotune.autotune_f0(self, f0, f0_autotune_strength)

        f0 *= pow(2, pitch / 12)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255

        return np.rint(f0_mel).astype(np.int32), f0.copy()

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        pitch_guidance = pitch != None and pitchf != None
        feats = torch.from_numpy(audio0).float()

        if feats.dim() == 2: feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        inputs = {"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12}

        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            if protect < 0.5 and pitch_guidance: feats0 = feats.clone()

            if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
                npy = feats[0].cpu().numpy()
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if protect < 0.5 and pitch_guidance: feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            p_len = audio0.shape[0] // self.window

            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch_guidance:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if protect < 0.5 and pitch_guidance:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = ((net_g.infer(feats, p_len, pitch if pitch_guidance else None, pitchf if pitch_guidance else None, sid)[0][0, 0]).data.cpu().float().numpy())

        del feats, p_len, padding_mask
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio1
    
    def pipeline(self, model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, filter_radius, tgt_sr, resample_sr, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts, audio_opt = [], []

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        t = None
        
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if pitch_guidance:
            pitch, pitchf = self.get_f0(audio_pad, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength)
            pitch, pitchf = pitch[:p_len], pitchf[:p_len]

            if self.device == "mps": pitchf = pitchf.astype(np.float32)
            pitch, pitchf = torch.tensor(pitch, device=self.device).unsqueeze(0).long(), torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])    
            s = t
            
        audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope)
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr: audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max

        if pitch_guidance: del pitch, pitchf
        del sid

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio_opt

class VoiceConverter:
    def __init__(self):
        self.config = config
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"
        self.checkpointing = False

    def load_embedders(self, embedder_model):
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([os.path.join("assets", "models", "embedders", embedder_model + '.pt')], suffix="")
        except Exception as e:
            logger.error(translations["read_model_error"].format(e=e))
        self.hubert_model = models[0].to(self.config.device).float().eval()

    def convert_audio(self, audio_input_path, audio_output_path, model_path, index_path, embedder_model, pitch, f0_method, index_rate, volume_envelope, protect, hop_length, f0_autotune, f0_autotune_strength, filter_radius, clean_audio, clean_strength, export_format, resample_sr = 0, sid = 0, checkpointing = False):
        try:
            self.get_vc(model_path, sid)
            audio = load_audio(audio_input_path)
            self.checkpointing = checkpointing

            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1: audio /= audio_max

            if not self.hubert_model: 
                if not os.path.exists(os.path.join("assets", "models", "embedders", embedder_model + '.pt')): raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")  
                self.load_embedders(embedder_model)

            if self.tgt_sr != resample_sr >= 16000: self.tgt_sr = resample_sr
            target_sr = min([8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000], key=lambda x: abs(x - self.tgt_sr))

            audio_output = self.vc.pipeline(model=self.hubert_model, net_g=self.net_g, sid=sid, audio=audio, pitch=pitch, f0_method=f0_method, file_index=(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")), index_rate=index_rate, pitch_guidance=self.use_f0, filter_radius=filter_radius, tgt_sr=self.tgt_sr, resample_sr=target_sr, volume_envelope=volume_envelope, version=self.version, protect=protect, hop_length=hop_length, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength)
            
            if clean_audio:
                from main.tools.noisereduce import reduce_noise
                audio_output = reduce_noise(y=audio_output, sr=target_sr, prop_decrease=clean_strength) 

            sf.write(audio_output_path, audio_output, target_sr, format=export_format)
        except Exception as e:
            logger.error(translations["error_convert"].format(e=e))

            import traceback
            logger.debug(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
          self.load_model(weight_root)
          if self.cpt is not None: self.setup()
          self.loaded_model = weight_root

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        self.cpt = (torch.load(weight_root, map_location="cpu") if os.path.isfile(weight_root) else None)

    def setup(self):
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

            self.use_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            self.vocoder = self.cpt.get("vocoder", "Default")

            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=self.text_enc_hidden_dim, vocoder=self.vocoder, checkpointing=self.checkpointing)
            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device).float()

            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]

if __name__ == "__main__": main()