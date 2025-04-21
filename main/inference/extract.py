import os
import re
import sys
import time
import tqdm
import torch
import shutil
import logging
import argparse
import warnings
import onnxruntime
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from random import shuffle
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.utils import check_predictors, check_embedders, load_audio, load_embedders_model

logger = logging.getLogger(__name__)
config = Config()
translations = config.translations
logger.propagate = False

warnings.filterwarnings("ignore")
for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "numba.core", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="-")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")

    return parser.parse_args()

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), config_save_path)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate):
    gt_wavs_dir, feature_dir = os.path.join(model_path, "sliced_audios"), os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir = None, None
    
    if pitch_guidance: f0_dir, f0nsf_dir = os.path.join(model_path, "f0"), os.path.join(model_path, "f0_voiced")

    gt_wavs_files, feature_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), set(name.split(".")[0] for name in os.listdir(feature_dir))
    names = gt_wavs_files & feature_files & set(name.split(".")[0] for name in os.listdir(f0_dir)) & set(name.split(".")[0] for name in os.listdir(f0nsf_dir)) if pitch_guidance else gt_wavs_files & feature_files

    options = []
    mute_base_path = os.path.join("assets", "logs", "mute")

    for name in names:
        options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0" if pitch_guidance else f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")

    mute_audio_path, mute_feature_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), os.path.join(mute_base_path, f"{rvc_version}_extracted", "mute.npy")
    for _ in range(2):
        options.append(f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|0" if pitch_guidance else f"{mute_audio_path}|{mute_feature_path}|0")

    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

def setup_paths(exp_dir, version = None):
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if version:
        out_path = os.path.join(exp_dir, f"{version}_extracted")
        os.makedirs(out_path, exist_ok=True)
        return wav_path, out_path
    else:
        output_root1, output_root2 = os.path.join(exp_dir, "f0"), os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True); os.makedirs(output_root2, exist_ok=True)
        return wav_path, output_root1, output_root2

def read_wave(wav_path, normalize = False, is_half = False):
    wav, sr = sf.read(wav_path, dtype=np.float32)
    assert sr == 16000, translations["sr_not_16000"]

    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2: feats = feats.mean(-1)
    feats = feats.view(1, -1)

    if normalize: feats = F.layer_norm(feats, feats.shape)
    return feats.half() if is_half else feats.float()

def get_device(gpu_index):
    try:
        index = int(gpu_index)
        if index < torch.cuda.device_count(): return f"cuda:{index}"
        else: logger.warning(translations["gpu_not_valid"])
    except ValueError:
        logger.warning(translations["gpu_not_valid"])
        return "cpu"

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers

class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160, is_half=False, device=config.device):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.is_half = is_half
    
    def compute_f0_hybrid(self, methods_str, np_arr, hop_length, f0_onnx):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack, resampled_stack = [], []
        logger.debug(translations["hybrid_methods"].format(methods=methods))

        for method in methods:
            f0 = None
            f0_methods = {"pm": lambda: self.get_pm(np_arr), "dio": lambda: self.get_pyworld(np_arr, "dio"), "mangio-crepe-full": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "full", onnx=f0_onnx), "mangio-crepe-large": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "large", onnx=f0_onnx), "mangio-crepe-medium": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "medium", onnx=f0_onnx), "mangio-crepe-small": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "small", onnx=f0_onnx), "mangio-crepe-tiny": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "tiny", onnx=f0_onnx), "crepe-full": lambda: self.get_crepe(np_arr, "full", onnx=f0_onnx), "crepe-large": lambda: self.get_crepe(np_arr, "large", onnx=f0_onnx), "crepe-medium": lambda: self.get_crepe(np_arr, "medium", onnx=f0_onnx), "crepe-small": lambda: self.get_crepe(np_arr, "small", onnx=f0_onnx), "crepe-tiny": lambda: self.get_crepe(np_arr, "tiny", onnx=f0_onnx), "fcpe": lambda: self.get_fcpe(np_arr, int(hop_length), onnx=f0_onnx), "fcpe-legacy": lambda: self.get_fcpe(np_arr, int(hop_length), legacy=True, onnx=f0_onnx), "rmvpe": lambda: self.get_rmvpe(np_arr, onnx=f0_onnx), "rmvpe-legacy": lambda: self.get_rmvpe(np_arr, legacy=True, onnx=f0_onnx), "harvest": lambda: self.get_pyworld(np_arr, "harvest"), "swipe": lambda: self.get_swipe(np_arr), "yin": lambda: self.get_yin(np_arr, int(hop_length), mode="yin"), "pyin": lambda: self.get_yin(np_arr, int(hop_length), mode="pyin")}
            f0 = f0_methods.get(method, lambda: ValueError(translations["method_not_valid"]))()
            f0_computation_stack.append(f0) 

        for f0 in f0_computation_stack:
            resampled_stack.append(np.interp(np.linspace(0, len(f0), (np_arr.size // self.hop)), np.arange(len(f0)), f0))

        return resampled_stack[0] if len(resampled_stack) == 1 else np.nanmedian(np.vstack(resampled_stack), axis=0)

    def compute_f0(self, np_arr, f0_method, hop_length, f0_onnx=False):
        f0_methods = {"pm": lambda: self.get_pm(np_arr), "dio": lambda: self.get_pyworld(np_arr, "dio"), "mangio-crepe-full": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "full", onnx=f0_onnx), "mangio-crepe-large": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "large", onnx=f0_onnx), "mangio-crepe-medium": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "medium", onnx=f0_onnx), "mangio-crepe-small": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "small", onnx=f0_onnx), "mangio-crepe-tiny": lambda: self.get_mangio_crepe(np_arr, int(hop_length), "tiny", onnx=f0_onnx), "crepe-full": lambda: self.get_crepe(np_arr, "full", onnx=f0_onnx), "crepe-large": lambda: self.get_crepe(np_arr, "large", onnx=f0_onnx), "crepe-medium": lambda: self.get_crepe(np_arr, "medium", onnx=f0_onnx), "crepe-small": lambda: self.get_crepe(np_arr, "small", onnx=f0_onnx), "crepe-tiny": lambda: self.get_crepe(np_arr, "tiny", onnx=f0_onnx), "fcpe": lambda: self.get_fcpe(np_arr, int(hop_length), onnx=f0_onnx), "fcpe-legacy": lambda: self.get_fcpe(np_arr, int(hop_length), legacy=True, onnx=f0_onnx), "rmvpe": lambda: self.get_rmvpe(np_arr, onnx=f0_onnx), "rmvpe-legacy": lambda: self.get_rmvpe(np_arr, legacy=True, onnx=f0_onnx), "harvest": lambda: self.get_pyworld(np_arr, "harvest"), "swipe": lambda: self.get_swipe(np_arr), "yin": lambda: self.get_yin(np_arr, int(hop_length), mode="yin"), "pyin": lambda: self.get_yin(np_arr, int(hop_length), mode="pyin")}
        return self.compute_f0_hybrid(f0_method, np_arr, int(hop_length), f0_onnx) if "hybrid" in f0_method else f0_methods.get(f0_method, lambda: ValueError(translations["method_not_valid"]))()

    def get_pm(self, x):
        import parselmouth

        f0 = (parselmouth.Sound(x, self.fs).to_pitch_ac(time_step=(160 / 16000 * 1000) / 1000, voicing_threshold=0.6, pitch_floor=50, pitch_ceiling=1100).selected_array["frequency"])
        pad_size = ((x.size // self.hop) - len(f0) + 1) // 2

        if pad_size > 0 or (x.size // self.hop) - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, (x.size // self.hop) - len(f0) - pad_size]], mode="constant")
        return f0
    
    def get_mangio_crepe(self, x, hop_length, model="full", onnx=False):
        from main.library.predictors.CREPE import predict

        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)

        source = predict(audio, self.fs, hop_length, self.f0_min, self.f0_max, model=model, batch_size=hop_length * 2, device=self.device, pad=True, providers=get_providers(), onnx=onnx).squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * (x.size // self.hop), len(source)) / (x.size // self.hop), np.arange(0, len(source)), source))
    
    def get_crepe(self, x, model="full", onnx=False):
        from main.library.predictors.CREPE import predict, mean, median

        f0, pd = predict(torch.tensor(np.copy(x))[None].float(), self.fs, 160, self.f0_min, self.f0_max, model, batch_size=512, device=self.device, return_periodicity=True, providers=get_providers(), onnx=onnx)
        f0, pd = mean(f0, 3), median(pd, 3)
        f0[pd < 0.1] = 0

        return f0[0].cpu().numpy()
    
    def get_fcpe(self, x, hop_length, legacy=False, onnx=False):
        from main.library.predictors.FCPE import FCPE

        model_fcpe = FCPE(os.path.join("assets", "models", "predictors", ("fcpe_legacy" if legacy else"fcpe") + (".onnx" if onnx else ".pt")), hop_length=int(hop_length), f0_min=int(self.f0_min), f0_max=int(self.f0_max), dtype=torch.float32, device=self.device, sample_rate=self.fs, threshold=0.03 if legacy else 0.006, providers=get_providers(), onnx=onnx, legacy=legacy, is_half=self.is_half)
        f0 = model_fcpe.compute_f0(x, p_len=(x.size // self.hop))

        del model_fcpe
        return f0
    
    def get_rmvpe(self, x, legacy=False, onnx=False):
        from main.library.predictors.RMVPE import RMVPE

        rmvpe_model = RMVPE(os.path.join("assets", "models", "predictors", "rmvpe" + (".onnx" if onnx else ".pt")), is_half=self.is_half, device=self.device, onnx=onnx, providers=get_providers())
        f0 = rmvpe_model.infer_from_audio_with_pitch(x, thred=0.03, f0_min=self.f0_min, f0_max=self.f0_max) if legacy else rmvpe_model.infer_from_audio(x, thred=0.03)

        del rmvpe_model
        return f0
    
    def get_pyworld(self, x, model="harvest"):
        from main.library.predictors.WORLD_WRAPPER import PYWORLD

        pw = PYWORLD()
        x = x.astype(np.double)

        if model == "harvest": f0, t = pw.harvest(x, fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        elif model == "dio": f0, t = pw.dio(x, fs=self.fs, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=1000 * self.hop / self.fs)
        else: raise ValueError(translations["method_not_valid"])

        return pw.stonemask(x, self.fs, t, f0)
    
    def get_swipe(self, x):
        from main.library.predictors.SWIPE import swipe

        f0, _ = swipe(x.astype(np.float32), self.fs, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=1000 * self.hop / self.fs)
        return f0
    
    def get_yin(self, x, hop_length, mode="yin"):
        import librosa

        if mode == "yin":
            source = np.array(librosa.yin(x.astype(np.float32), sr=self.fs, fmin=self.f0_min, fmax=self.f0_max, hop_length=hop_length))
            source[source < 0.001] = np.nan
        else:
            f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=self.f0_min, fmax=self.f0_max, sr=self.fs, hop_length=hop_length)

            source = np.array(f0)
            source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * (x.size // self.hop), len(source)) / (x.size // self.hop), np.arange(0, len(source)), source))
    
    def coarse_f0(self, f0):
        return np.rint(np.clip(((1127 * np.log(1 + f0 / 700)) - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1, 1, self.f0_bin - 1)).astype(int)

    def process_file(self, file_info, f0_method, hop_length, f0_onnx):
        inp_path, opt_path1, opt_path2, np_arr = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length, f0_onnx)
            if isinstance(feature_pit, tuple): feature_pit = feature_pit[0]

            np.save(opt_path2, feature_pit, allow_pickle=False)
            np.save(opt_path1, self.coarse_f0(feature_pit), allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"{translations['extract_file_error']} {inp_path}: {e}")

    def process_files(self, files, f0_method, hop_length, f0_onnx, pbar):
        for file_info in files:
            self.process_file(file_info, f0_method, hop_length, f0_onnx)
            pbar.update()

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus, f0_onnx, is_half):
    input_root, *output_roots = setup_paths(exp_dir)
    output_root1, output_root2 = output_roots if len(output_roots) == 2 else (output_roots[0], None)

    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, load_audio(logger, os.path.join(input_root, name), 16000)) for name in sorted(os.listdir(input_root)) if "spec" not in name]
    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))

    start_time = time.time()
    gpus = gpus.split("-")
    process_partials = []

    pbar = tqdm.tqdm(total=len(paths), ncols=100, unit="p")
    for idx, gpu in enumerate(gpus):
        feature_input = FeatureInput(device=get_device(gpu) if gpu != "" else "cpu", is_half=is_half)
        process_partials.append((feature_input, paths[idx::len(gpus)]))

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        for future in as_completed([executor.submit(FeatureInput.process_files, feature_input, part_paths, f0_method, hop_length, f0_onnx, pbar) for feature_input, part_paths in process_partials]):
            pbar.update(1)
            logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))
            future.result()

    pbar.close()
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def extract_features(model, feats, version):
    return torch.as_tensor(model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {"feats": feats.detach().cpu().numpy()})[0 if version == "v1" else 1], dtype=torch.float32, device=feats.device)

def process_file_embedding(file, wav_path, out_path, model, device, version, saved_cfg, embed_suffix, is_half):
    out_file_path = os.path.join(out_path, file.replace("wav", "npy"))
    if os.path.exists(out_file_path): return
    feats = read_wave(os.path.join(wav_path, file), normalize=saved_cfg.task.normalize if saved_cfg else False, is_half=is_half).to(device)

    with torch.no_grad():
        if embed_suffix == ".pt":
            model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()
            logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9 if version == "v1" else 12})
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        elif embed_suffix == ".onnx": feats = extract_features(model, feats, version).to(device)
        elif embed_suffix == ".safetensors":
            model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()
            logits = model(feats)["last_hidden_state"]
            feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
        else: raise ValueError(translations["option_not_valid"])

    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
    else: logger.warning(f"{file} {translations['NaN']}")

def run_embedding_extraction(exp_dir, version, gpus, embedder_model, embedders_mode, is_half):
    wav_path, out_path = setup_paths(exp_dir, version)
    logger.info(translations["start_extract_hubert"])

    start_time = time.time()
    models, saved_cfg, embed_suffix = load_embedders_model(embedder_model, embedders_mode, providers=get_providers())
    devices = [get_device(gpu) for gpu in (gpus.split("-") if gpus != "-" else ["cpu"])]
    paths = sorted([file for file in os.listdir(wav_path) if file.endswith(".wav")])

    if not paths:
        logger.warning(translations["not_found_audio_file"])
        sys.exit(1)

    pbar = tqdm.tqdm(total=len(paths) * len(devices), ncols=100, unit="p")
    for task in [(file, wav_path, out_path, models, device, version, saved_cfg, embed_suffix, is_half) for file in paths for device in devices]:
        try:
            process_file_embedding(*task)
        except Exception as e:
            raise RuntimeError(f"{translations['process_error']} {task[0]}: {e}")
        
        pbar.update(1)
        logger.debug(pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"]))

    pbar.close()
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def main():
    args = parse_arguments()
    exp_dir = os.path.join("assets", "logs", args.model_name)
    f0_method, hop_length, num_processes, gpus, version, pitch_guidance, sample_rate, embedder_model, f0_onnx, embedders_mode = args.f0_method, args.hop_length, args.cpu_cores, args.gpu, args.rvc_version, args.pitch_guidance, args.sample_rate, args.embedder_model, args.f0_onnx, args.embedders_mode

    check_predictors(f0_method, f0_onnx); check_embedders(embedder_model, embedders_mode)
    if logger.hasHandlers(): logger.handlers.clear()
    else:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        file_handler = logging.handlers.RotatingFileHandler(os.path.join(exp_dir, "extract.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    log_data = {translations['modelname']: args.model_name, translations['export_process']: exp_dir, translations['f0_method']: f0_method, translations['pretrain_sr']: sample_rate, translations['cpu_core']: num_processes, "Gpu": gpus, "Hop length": hop_length, translations['training_version']: version, translations['extract_f0']: pitch_guidance, translations['hubert_model']: embedder_model, translations["f0_onnx_mode"]: f0_onnx, translations["embed_mode"]: embedders_mode}
    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, gpus, f0_onnx, config.is_half)
        run_embedding_extraction(exp_dir, version, gpus, embedder_model, embedders_mode, config.is_half)
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate)
    except Exception as e:
        logger.error(f"{translations['extract_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")

if __name__ == "__main__": main()