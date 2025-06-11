import os
import sys
import time
import shutil
import codecs
import threading
import subprocess

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs

def if_done(done, p):
    while 1:
        if p.poll() is None: time.sleep(0.5)
        else: break

    done[0] = True

def log_read(done, name):
    log_file = os.path.join(configs["logs_path"], "app.log")

    f = open(log_file, "w", encoding="utf-8")
    f.close()

    while 1:
        with open(log_file, "r", encoding="utf-8") as f:
            yield "".join(line for line in f.readlines() if "DEBUG" not in line and name in line and line.strip() != "")

        time.sleep(1)
        if done[0]: break

    with open(log_file, "r", encoding="utf-8") as f:
        log = "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

    yield log

def create_dataset(input_audio, output_dataset, clean_dataset, clean_strength, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size, sample_rate):
    version = 1 if kim_vocals_version == "Version-1" else 2

    gr_info(translations["start"].format(start=translations["create"]))

    p = subprocess.Popen(f'{python} {configs["create_dataset_path"]} --input_audio "{input_audio}" --output_dataset "{output_dataset}" --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}" --sample_rate {sample_rate}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "create_dataset"):
        yield log

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, dataset, clean_dataset, clean_strength):
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    if not os.path.exists(dataset) or not any(f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))): return gr_warning(translations["not_found_data"])
    
    model_dir = os.path.join(configs["logs_path"], model_name)
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    p = subprocess.Popen(f'{python} {configs["preprocess_path"]} --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "preprocess"):
        yield log

def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders, onnx_f0_mode, embedders_mode, f0_autotune, f0_autotune_strength, hybrid_method):
    f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)

    try:
        if not any(os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios"))) or not any(os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))): return gr_warning(translations["not_found_data_preprocess"])
    except:
        return gr_warning(translations["not_found_data_preprocess"])
    
    p = subprocess.Popen(f'{python} {configs["extract_path"]} --model_name "{model_name}" --rvc_version {version} --f0_method {f0method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model} --f0_onnx {onnx_f0_mode} --embedders_mode {embedders_mode} --f0_autotune {f0_autotune} --f0_autotune_strength {f0_autotune_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "extract"):
        yield log

def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join(configs["logs_path"], model_name)
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    p = subprocess.Popen(f'{python} {configs["create_index_path"]} --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(done, "create_index"):
        yield log

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing, deterministic, benchmark, optimizer):
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join(configs["logs_path"], model_name)
    if os.path.exists(os.path.join(model_dir, "train_pid.txt")): os.remove(os.path.join(model_dir, "train_pid.txt"))
    
    try:
        if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])
    except:
        return gr_warning(translations["not_found_data_extract"])
    
    if not not_pretrain:
        if not custom_pretrained: 
            pretrained_selector = {True: {32000: ("f0G32k.pth", "f0D32k.pth"), 40000: ("f0G40k.pth", "f0D40k.pth"), 48000: ("f0G48k.pth", "f0D48k.pth")}, False: {32000: ("G32k.pth", "D32k.pth"), 40000: ("G40k.pth", "D40k.pth"), 48000: ("G48k.pth", "D48k.pth")}}

            pg, pd = pretrained_selector[pitch_guidance][sr]
        else:
            if not pretrain_g: return gr_warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg, pd = pretrain_g, pretrain_d

        pretrained_G, pretrained_D = (os.path.join(configs["pretrained_v2_path"] if rvc_version == 'v2' else configs["pretrained_v1_path"], f"{vocoder}_{pg}" if vocoder != 'Default' else pg), os.path.join(configs["pretrained_v2_path"] if rvc_version == 'v2' else configs["pretrained_v1_path"], f"{vocoder}_{pd}" if vocoder != 'Default' else pd)) if not custom_pretrained else (os.path.join(configs["pretrained_custom_path"], pg), os.path.join(configs["pretrained_custom_path"], pd))
        download_version = codecs.decode(f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_i{'2' if rvc_version == 'v2' else '1'}/", "rot13")
        
        if not custom_pretrained:
            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pg]) if vocoder != 'Default' else (download_version + pg), os.path.join(configs["pretrained_v2_path"] if rvc_version == 'v2' else configs["pretrained_v1_path"], f"{vocoder}_{pg}" if vocoder != 'Default' else pg))
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pd]) if vocoder != 'Default' else (download_version + pd), os.path.join(configs["pretrained_v2_path"] if rvc_version == 'v2' else configs["pretrained_v1_path"], f"{vocoder}_{pd}" if vocoder != 'Default' else pd))
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G = pretrained_D = None
        else:
            if not os.path.exists(pretrained_G): return gr_warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: 
        pretrained_G = pretrained_D = None
        gr_warning(translations["not_use_pretrain"])

    gr_info(translations["start"].format(start=translations["training"]))

    p = subprocess.Popen(f'{python} {configs["train_path"]} --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --cleanup {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder "{vocoder}" --checkpointing {checkpointing} --deterministic {deterministic} --benchmark {benchmark} --optimizer {optimizer}', shell=True)
    done = [False]

    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(done, "train"):
        lines = log.splitlines()
        if len(lines) > 100: log = "\n".join(lines[-100:])
        yield log