import os
import re
import sys
import json
import torch
import shutil

import gradio as gr

sys.path.append(os.getcwd())

from main.library import torch_amd
from main.app.variables import config, configs, configs_json, logger, translations, edgetts, google_tts_voice, method_f0, method_f0_full

def gr_info(message):
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = [f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" for i in range(ngpu) if torch.cuda.is_available() or ngpu != 0]

    if len(gpu_infos) == 0:
        ngpu = torch_amd.device_count()
        gpu_infos = [f"{i}: {torch_amd.device_name(i)}" for i in range(ngpu) if torch_amd.is_available() or ngpu != 0]

    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]

def gpu_number_str():
    ngpu = torch.cuda.device_count()
    if ngpu == 0: ngpu = torch_amd.device_count()

    return str("-".join(map(str, range(ngpu))) if torch.cuda.is_available() or torch_amd.is_available() else "-")

def change_f0_choices(): 
    f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["f0_path"]) for f in files if f.endswith(".txt")])
    return {"value": f0_file[0] if len(f0_file) >= 1 else "", "choices": f0_file, "__type__": "update"}

def change_audios_choices(input_audio): 
    audios = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(configs["audios_path"]) for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), "choices": audios, "__type__": "update"}

def change_models_choices():
    model, index = sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name])
    return [{"value": model[0] if len(model) >= 1 else "", "choices": model, "__type__": "update"}, {"value": index[0] if len(index) >= 1 else "", "choices": index, "__type__": "update"}]

def change_pretrained_choices():
    return [{"choices": sorted([model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "D" in model]), "__type__": "update"}, {"choices": sorted([model for model in os.listdir(configs["pretrained_custom_path"]) if model.endswith(".pth") and "G" in model]), "__type__": "update"}]

def change_choices_del():
    return [{"choices": sorted(list(model for model in os.listdir(configs["weights_path"]) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join(configs["logs_path"], f) for f in os.listdir(configs["logs_path"]) if "mute" not in f and os.path.isdir(os.path.join(configs["logs_path"], f))]), "__type__": "update"}]

def change_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".conversion.json"))), "__type__": "update"}

def change_effect_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(configs["presets_path"]) if f.endswith(".effect.json"))), "__type__": "update"}

def change_tts_voice_choices(google):
    return {"choices": google_tts_voice if google else edgetts, "value": google_tts_voice[0] if google else edgetts[0], "__type__": "update"}

def change_backing_choices(backing, merge):
    if backing or merge: return {"value": False, "interactive": False, "__type__": "update"}
    elif not backing or not merge: return  {"interactive": True, "__type__": "update"}
    else: gr_warning(translations["option_not_valid"])

def change_download_choices(select):
    selects = [False]*10

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["download_from_csv"]:  selects[3] = selects[4] = True
    elif select == translations["search_models"]: selects[5] = selects[6] = True
    elif select == translations["upload"]: selects[9] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def change_download_pretrained_choices(select):
    selects = [False]*8

    if select == translations["download_url"]: selects[0] = selects[1] = selects[2] = True
    elif select == translations["list_model"]: selects[3] = selects[4] = selects[5] = True
    elif select == translations["upload"]: selects[6] = selects[7] = True
    else: gr_warning(translations["option_not_valid"])

    return [{"visible": selects[i], "__type__": "update"} for i in range(len(selects))]

def get_index(model):
    model = os.path.basename(model).split("_")[0]
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(configs["logs_path"], topdown=False) for name in files if name.endswith(".index") and "trained" not in name] if model.split(".")[0] in f), ""), "__type__": "update"} if model else None

def index_strength_show(index):
    return {"visible": index != "" and os.path.exists(index), "value": 0.5, "__type__": "update"}

def hoplength_show(method, hybrid_method=None):
    show_hop_length_method = ["mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "fcpe", "fcpe-legacy", "yin", "pyin"]

    if method in show_hop_length_method: visible = True
    elif method == "hybrid":
        methods_str = re.search("hybrid\[(.+)\]", hybrid_method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        for i in methods:
            visible = i in show_hop_length_method
            if visible: break
    else: visible = False
    
    return {"visible": visible, "__type__": "update"}

def visible(value):
    return {"visible": value, "__type__": "update"}

def valueFalse_interactive(value): 
    return {"value": False, "interactive": value, "__type__": "update"}

def valueEmpty_visible1(value): 
    return {"value": "", "visible": value, "__type__": "update"}

def pitch_guidance_lock(vocoders):
    return {"value": True, "interactive": vocoders == "Default", "__type__": "update"}

def vocoders_lock(pitch, vocoders):
    return {"value": vocoders if pitch else "Default", "interactive": pitch, "__type__": "update"}

def unlock_f0(value):
    return {"choices": method_f0_full if value else method_f0, "value": "rmvpe", "__type__": "update"} 

def unlock_vocoder(value, vocoder):
    return {"value": vocoder if value == "v2" else "Default", "interactive": value == "v2", "__type__": "update"} 

def unlock_ver(value, vocoder):
    return {"value": "v2" if vocoder == "Default" else value, "interactive": vocoder == "Default", "__type__": "update"}

def visible_embedders(value):
    return {"visible": value != "spin", "__type__": "update"}

def change_fp(fp):
    fp16 = fp == "fp16"

    if fp16 and config.device in ["cpu", "mps", "ocl:0"]: 
        gr_warning(translations["fp16_not_support"])
        return "fp32"
    else:
        gr_info(translations["start_update_precision"])

        configs = json.load(open(configs_json, "r"))
        configs["fp16"] = config.is_half = fp16

        with open(configs_json, "w") as f:
            json.dump(configs, f, indent=4)

        gr_info(translations["success"])
        return "fp16" if fp16 else "fp32"
    
def process_output(file_path):
    if config.configs.get("delete_exists_file", True):
        if os.path.exists(file_path): os.remove(file_path)
        return file_path
    else:
        if not os.path.exists(file_path): return file_path
        file = os.path.basename(file_path).split(".")

        index = 1
        while 1:
            file_path = os.path.join(os.path.dirname(file_path), f"{file[0]}_{index}." + file[1])
            if not os.path.exists(file_path): return file_path
            index += 1

def shutil_move(input_path, output_path):
    output_path = os.path.join(output_path, os.path.basename(input_path)) if os.path.isdir(output_path) else output_path

    return shutil.move(input_path, process_output(output_path)) if os.path.exists(output_path) else shutil.move(input_path, output_path)