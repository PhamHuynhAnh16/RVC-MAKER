import os
import re
import ssl
import sys
import json
import torch
import codecs
import shutil
import asyncio
import librosa
import logging
import datetime
import platform
import requests
import warnings
import threading
import subprocess
import logging.handlers

import numpy as np
import gradio as gr
import pandas as pd
import soundfile as sf

from time import sleep
from multiprocessing import cpu_count

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.configs.config import Config

ssl._create_default_https_context = ssl._create_unverified_context
logger = logging.getLogger(__name__)
logger.propagate = False

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join("assets", "logs", "app.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")
for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

config = Config()
python = sys.executable
translations = config.translations 
configs_json = os.path.join("main", "configs", "config.json")
configs = json.load(open(configs_json, "r"))

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

if config.device in ["cpu", "mps"]  and configs.get("fp16", False):
    logger.warning(translations["fp16_not_support"])
    configs["fp16"] = config.is_half = False
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

models, model_options = {}, {}

method_f0 = ["mangio-crepe-full", "crepe-full", "fcpe", "rmvpe", "harvest", "pyin"]
method_f0_full = ["pm", "dio", "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full", "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full", "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy", "harvest", "yin", "pyin", "swipe"]

embedders_mode = ["fairseq", "onnx", "transformers", "spin"]
embedders_model = ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base", "custom"]

paths_for_files = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
model_name, index_path, delete_index = sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name]), sorted([os.path.join("assets", "logs", f) for f in os.listdir(os.path.join("assets", "logs")) if "mute" not in f and os.path.isdir(os.path.join("assets", "logs", f))])
pretrainedD, pretrainedG, Allpretrained = ([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model], [model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model], [os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)])

separate_model = sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if models.endswith((".th", ".yaml", ".onnx"))])
presets_file = sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json")))
f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(os.path.join("assets", "f0")) for f in files if f.endswith(".txt")])

language, theme, edgetts, google_tts_voice, mdx_model, uvr_model, font = configs.get("language", "vi-VN"), configs.get("theme", "NoCrypt/miku"), configs.get("edge_tts", ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]), configs.get("google_tts_voice", ["vi", "en"]), configs.get("mdx_model", "MDXNET_Main"), (configs.get("demucs_model", "HD_MMI") + configs.get("mdx_model", "MDXNET_Main")), configs.get("font", "https://fonts.googleapis.com/css2?family=Courgette&display=swap")

csv_path = os.path.join("assets", "spreadsheet.csv")
logger.info(config.device)

if "--allow_all_disk" in sys.argv:
    import win32api

    allow_disk = win32api.GetLogicalDriveStrings().split('\x00')[:-1]
else: allow_disk = []

if language == "vi-VN": 
    import gradio.strings
    gradio.strings.en = {"RUNNING_LOCALLY": "* Chạy trên liên kết nội bộ:  {}://{}:{}", "RUNNING_LOCALLY_SSR": "* Chạy trên liên kết nội bộ:  {}://{}:{}, với SSR ⚡ (thử nghiệm, để tắt hãy dùng `ssr=False` trong `launch()`)", "SHARE_LINK_DISPLAY": "* Chạy trên liên kết công khai: {}", "COULD_NOT_GET_SHARE_LINK": "\nKhông thể tạo liên kết công khai. Vui lòng kiểm tra kết nối mạng của bạn hoặc trang trạng thái của chúng tôi: https://status.gradio.app.", "COULD_NOT_GET_SHARE_LINK_MISSING_FILE": "\nKhông thể tạo liên kết công khai. Thiếu tập tin: {}. \n\nVui lòng kiểm tra kết nối internet của bạn. Điều này có thể xảy ra nếu phần mềm chống vi-rút của bạn chặn việc tải xuống tệp này. Bạn có thể cài đặt thủ công bằng cách làm theo các bước sau: \n\n1. Tải xuống tệp này: {}\n2. Đổi tên tệp đã tải xuống thành: {}\n3. Di chuyển tệp đến vị trí này: {}", "COLAB_NO_LOCAL": "Không thể hiển thị giao diện nội bộ trên google colab, liên kết công khai đã được tạo.", "PUBLIC_SHARE_TRUE": "\nĐể tạo một liên kết công khai, hãy đặt `share=True` trong `launch()`.", "MODEL_PUBLICLY_AVAILABLE_URL": "Mô hình được cung cấp công khai tại: {} (có thể mất tới một phút để sử dụng được liên kết)", "GENERATING_PUBLIC_LINK": "Đang tạo liên kết công khai (có thể mất vài giây...):", "BETA_INVITE": "\nCảm ơn bạn đã là người dùng Gradio! Nếu bạn có thắc mắc hoặc phản hồi, vui lòng tham gia máy chủ Discord của chúng tôi và trò chuyện với chúng tôi: https://discord.gg/feTf9x3ZSB", "COLAB_DEBUG_TRUE": "Đã phát hiện thấy sổ tay Colab. Ô này sẽ chạy vô thời hạn để bạn có thể xem lỗi và nhật ký. " "Để tắt, hãy đặt debug=False trong launch().", "COLAB_DEBUG_FALSE": "Đã phát hiện thấy sổ tay Colab. Để hiển thị lỗi trong sổ ghi chép colab, hãy đặt debug=True trong launch()", "COLAB_WARNING": "Lưu ý: việc mở Chrome Inspector có thể làm hỏng bản demo trong sổ tay Colab.", "SHARE_LINK_MESSAGE": "\nLiên kết công khai sẽ hết hạn sau 72 giờ. Để nâng cấp GPU và lưu trữ vĩnh viễn miễn phí, hãy chạy `gradio deploy` từ terminal trong thư mục làm việc để triển khai lên huggingface (https://huggingface.co/spaces)", "INLINE_DISPLAY_BELOW": "Đang tải giao diện bên dưới...", "COULD_NOT_GET_SHARE_LINK_CHECKSUM": "\nKhông thể tạo liên kết công khai. Tổng kiểm tra không khớp cho tập tin: {}."}

if os.path.exists(csv_path): cached_data = pd.read_csv(csv_path) 
else:
    cached_data = pd.read_csv(codecs.decode("uggcf://qbpf.tbbtyr.pbz/fcernqfurrgf/q/1gNHnDeRULtEfz1Yieaw14USUQjWJy0Oq9k0DrCrjApb/rkcbeg?sbezng=pfi&tvq=1977693859", "rot13"))
    cached_data.to_csv(csv_path, index=False)

for _, row in cached_data.iterrows():
    filename = row['Filename']
    url = None

    for value in row.values:
        if isinstance(value, str) and "huggingface" in value:
            url = value
            break

    if url: models[filename] = url



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
    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]

def change_f0_choices(): 
    f0_file = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk(os.path.join("assets", "f0")) for f in files if f.endswith(".txt")])
    return {"value": f0_file[0] if len(f0_file) >= 1 else "", "choices": f0_file, "__type__": "update"}

def change_audios_choices(input_audio): 
    audios = sorted([os.path.abspath(os.path.join(root, f)) for root, _, files in os.walk("audios") for f in files if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3")])
    return {"value": input_audio if input_audio != "" else (audios[0] if len(audios) >= 1 else ""), "choices": audios, "__type__": "update"}

def change_separate_choices():
    return [{"choices": sorted([os.path.join("assets", "models", "uvr5", models) for models in os.listdir(os.path.join("assets", "models", "uvr5")) if model.endswith((".th", ".yaml", ".onnx"))]), "__type__": "update"}]

def change_models_choices():
    model, index = sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith((".pth", ".onnx")) and not model.startswith("G_") and not model.startswith("D_"))), sorted([os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name])
    return [{"value": model[0] if len(model) >= 1 else "", "choices": model, "__type__": "update"}, {"value": index[0] if len(index) >= 1 else "", "choices": index, "__type__": "update"}]

def change_allpretrained_choices():
    return [{"choices": sorted([os.path.join("assets", "models", path, model) for path in ["pretrained_v1", "pretrained_v2", "pretrained_custom"] for model in os.listdir(os.path.join("assets", "models", path)) if model.endswith(".pth") and ("D" in model or "G" in model)]), "__type__": "update"}]

def change_pretrained_choices():
    return [{"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "D" in model]), "__type__": "update"}, {"choices": sorted([model for model in os.listdir(os.path.join("assets", "models", "pretrained_custom")) if model.endswith(".pth") and "G" in model]), "__type__": "update"}]

def change_choices_del():
    return [{"choices": sorted(list(model for model in os.listdir(os.path.join("assets", "weights")) if model.endswith(".pth") and not model.startswith("G_") and not model.startswith("D_"))), "__type__": "update"}, {"choices": sorted([os.path.join("assets", "logs", f) for f in os.listdir(os.path.join("assets", "logs")) if "mute" not in f and os.path.isdir(os.path.join("assets", "logs", f))]), "__type__": "update"}]

def change_preset_choices():
    return {"value": "", "choices": sorted(list(f for f in os.listdir(os.path.join("assets", "presets")) if f.endswith(".json"))), "__type__": "update"}

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
    return {"value": next((f for f in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".index") and "trained" not in name] if model.split(".")[0] in f), ""), "__type__": "update"} if model else None

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

def valueFalse_interactive(inp): 
    return {"value": False, "interactive": inp, "__type__": "update"}

def valueEmpty_visible1(inp1): 
    return {"value": "", "visible": inp1, "__type__": "update"}

def process_input(file_path):
    file_contents = ""

    if not file_path.endswith(".srt"):
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()

    gr_info(translations["upload_success"].format(name=translations["text"]))
    return file_contents

def fetch_pretrained_data():
    response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/wfba/phfgbz_cergenvarq.wfba", "rot13"))
    response.raise_for_status()

    return response.json()

def update_sample_rate_dropdown(model):
    data = fetch_pretrained_data()
    if model != translations["success"]: return {"choices": list(data[model].keys()), "value": list(data[model].keys())[0], "__type__": "update"}

def if_done(done, p):
    while 1:
        if p.poll() is None: sleep(0.5)
        else: break

    done[0] = True

def restart_app():
    global app

    gr_info(translations["15s"])
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    app.close()
    subprocess.run([python, os.path.join("main", "app", "app.py")] + sys.argv[1:])

def change_language(lang):
    configs = json.load(open(configs_json, "r"))
    configs["language"] = lang

    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def change_theme(theme):
    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["theme"] = theme
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def change_font(font):
    with open(configs_json, "r") as f:
        configs = json.load(f)

    configs["font"] = font
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    restart_app()

def zip_file(name, pth, index):
    pth_path = os.path.join("assets", "weights", pth)
    if not pth or not os.path.exists(pth_path) or not pth.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))

    zip_file_path = os.path.join("assets", "logs", name, name + ".zip")
    gr_info(translations["start"].format(start=translations["zip"]))

    import zipfile
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        if index: zipf.write(index, os.path.basename(index))

    gr_info(translations["success"])
    return {"visible": True, "value": zip_file_path, "__type__": "update"}

def fetch_models_data(search):
    all_table_data = [] 
    page = 1 

    while 1:
        try:
            response = requests.post(url=codecs.decode("uggcf://ibvpr-zbqryf.pbz/srgpu_qngn.cuc", "rot13"), data={"page": page, "search": search})

            if response.status_code == 200:
                table_data = response.json().get("table", "")
                if not table_data.strip(): break  
                all_table_data.append(table_data)
                page += 1
            else:
                logger.debug(f"{translations['code_error']} {response.status_code}")
                break  
        except json.JSONDecodeError:
            logger.debug(translations["json_error"])
            break
        except requests.RequestException as e:
            logger.debug(translations["requests_error"].format(e=e))
            break
    return all_table_data

def search_models(name):
    gr_info(translations["start"].format(start=translations["search"]))
    tables = fetch_models_data(name)

    if len(tables) == 0:
        gr_info(translations["not_found"].format(name=name))
        return [None]*2
    else:
        model_options.clear()
        
        from bs4 import BeautifulSoup

        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag, url_tag = row.find("a", {"class": "fs-5"}), row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                url = url_tag["href"].replace("https://easyaivoice.com/run?url=", "")
                if "huggingface" in url:
                    if name_tag and url_tag: model_options[name_tag.text.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()] = url

        gr_info(translations["found"].format(results=len(model_options)))
        return [{"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"}, {"value": translations["downloads"], "visible": True, "__type__": "update"}]

def move_files_from_directory(src_dir, dest_weights, dest_logs, model_name):
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                model_log_dir = os.path.join(dest_logs, model_name)
                os.makedirs(model_log_dir, exist_ok=True)

                filepath = os.path.join(model_log_dir, file.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip())
                if os.path.exists(filepath): os.remove(filepath)

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_weights, model_name + ".pth")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)
            elif file.endswith(".onnx") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_weights, model_name + ".onnx")
                if os.path.exists(pth_path): os.remove(pth_path)

                shutil.move(file_path, pth_path)

def download_url(url):
    import yt_dlp

    if not url: return gr_warning(translations["provide_url"])
    if not os.path.exists("audios"): os.makedirs("audios", exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }],
            "quiet": True,            
            "no_warnings": True,
            "noplaylist": True,
            "verbose": False,
            "cookiefile": "assets/yt-dlp/config.txt"
        }
        gr_info(translations["start"].format(start=translations["download_music"]))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join("audios", re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))
            if os.path.exists(audio_output): shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = audio_output + ".wav"
            if os.path.exists(audio_output): os.remove(audio_output)
            
            ydl.download([url])

        gr_info(translations["success"])
        return [audio_output, audio_output, translations["success"]]

def download_model(url=None, model=None):
    if not url: return gr_warning(translations["provide_url"])
    if not model: return gr_warning(translations["provide_name_is_save"])

    model = model.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    download_dir = os.path.join("download_model")
    weights_dir = os.path.join("assets", "weights")
    logs_dir = os.path.join("assets", "logs")

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)
    
    try:
        gr_info(translations["start"].format(start=translations["download"]))

        if url.endswith(".pth"): huggingface.HF_download_file(url, os.path.join(weights_dir, f"{model}.pth"))
        elif url.endswith(".onnx"): huggingface.HF_download_file(url, os.path.join(weights_dir, f"{model}.onnx"))
        elif url.endswith(".index"):
            model_log_dir = os.path.join(logs_dir, model)
            os.makedirs(model_log_dir, exist_ok=True)

            huggingface.HF_download_file(url, os.path.join(model_log_dir, f"{model}.index"))
        elif url.endswith(".zip"):
            output_path = huggingface.HF_download_file(url, os.path.join(download_dir, model + ".zip"))
            shutil.unpack_archive(output_path, download_dir)

            move_files_from_directory(download_dir, weights_dir, logs_dir, model)
        else:
            if "drive.google.com" in url or "drive.usercontent.google.com" in url:
                file_id = None

                from main.tools import gdown

                if "/file/d/" in url: file_id = url.split("/d/")[1].split("/")[0]
                elif "open?id=" in url: file_id = url.split("open?id=")[1].split("/")[0]
                elif "/download?id=" in url: file_id = url.split("/download?id=")[1].split("&")[0]
                
                if file_id:
                    file = gdown.gdown_download(id=file_id, output=download_dir)
                    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                    move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mega.nz" in url:
                from main.tools import meganz
                
                meganz.mega_download_url(url, download_dir)

                file_download = next((f for f in os.listdir(download_dir)), None)
                if file_download.endswith(".zip"): shutil.unpack_archive(os.path.join(download_dir, file_download), download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "mediafire.com" in url:
                from main.tools import mediafire

                file = mediafire.Mediafire_Download(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            elif "pixeldrain.com" in url:
                from main.tools import pixeldrain

                file = pixeldrain.pixeldrain(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, weights_dir, logs_dir, model)
            else:
                gr_warning(translations["not_support_url"])
                return translations["not_support_url"]
        
        gr_info(translations["success"])
        return translations["success"]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return translations["error_occurred"].format(e=e)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

def save_drop_model(dropbox):
    weight_folder = os.path.join("assets", "weights")
    logs_folder = os.path.join("assets", "logs")
    save_model_temp = os.path.join("save_model_temp")

    if not os.path.exists(weight_folder): os.makedirs(weight_folder, exist_ok=True)
    if not os.path.exists(logs_folder): os.makedirs(logs_folder, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    shutil.move(dropbox, save_model_temp)

    try:
        file_name = os.path.basename(dropbox)

        if file_name.endswith(".pth") and file_name.endswith(".onnx") and file_name.endswith(".index"): gr_warning(translations["not_model"])
        else:    
            if file_name.endswith(".zip"):
                shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
                move_files_from_directory(save_model_temp, weight_folder, logs_folder, file_name.replace(".zip", ""))
            elif file_name.endswith((".pth", ".onnx")): 
                output_file = os.path.join(weight_folder, file_name)
                if os.path.exists(output_file): os.remove(output_file)
                
                shutil.move(os.path.join(save_model_temp, file_name), output_file)
            elif file_name.endswith(".index"):
                def extract_name_model(filename):
                    match = re.search(r"([A-Za-z]+)(?=_v|\.|$)", filename)
                    return match.group(1) if match else None
                
                model_logs = os.path.join(logs_folder, extract_name_model(file_name))
                if not os.path.exists(model_logs): os.makedirs(model_logs, exist_ok=True)
                shutil.move(os.path.join(save_model_temp, file_name), model_logs)
            else: 
                gr_warning(translations["unable_analyze_model"])
                return None
        
        gr_info(translations["upload_success"].format(name=translations["model"]))
        return None
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return None
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)

def download_pretrained_model(choices, model, sample_rate):
    pretraineds_custom_path = os.path.join("assets", "models", "pretrained_custom")
    if choices == translations["list_model"]:
        paths = fetch_pretrained_data()[model][sample_rate]

        if not os.path.exists(pretraineds_custom_path): os.makedirs(pretraineds_custom_path, exist_ok=True)
        url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_phfgbz/", "rot13") + paths

        gr_info(translations["download_pretrain"])
        file = huggingface.HF_download_file(url.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), os.path.join(pretraineds_custom_path, paths))

        if file.endswith(".zip"): 
            shutil.unpack_archive(file, pretraineds_custom_path)
            os.remove(file)

        gr_info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        if not model: return gr_warning(translations["provide_pretrain"].format(dg="D"))
        if not sample_rate: return gr_warning(translations["provide_pretrain"].format(dg="G"))

        gr_info(translations["download_pretrain"])

        huggingface.HF_download_file(model.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)
        huggingface.HF_download_file(sample_rate.replace("/blob/", "/resolve/").replace("?download=true", "").strip(), pretraineds_custom_path)

        gr_info(translations["success"])
        return translations["success"]

def fushion_model_pth(name, pth_1, pth_2, ratio):
    if not name.endswith(".pth"): name = name + ".pth"

    if not pth_1 or not os.path.exists(pth_1) or not pth_1.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 1"))
        return [translations["provide_file"].format(filename=translations["model"] + " 1"), None]
    
    if not pth_2 or not os.path.exists(pth_2) or not pth_2.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"] + " 2"))
        return [translations["provide_file"].format(filename=translations["model"] + " 2"), None]
    
    from collections import OrderedDict

    def extract(ckpt):
        a = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}

        for key in a.keys():
            if "enc_q" in key: continue

            opt["weight"][key] = a[key]

        return opt
    
    try:
        ckpt1 = torch.load(pth_1, map_location="cpu")
        ckpt2 = torch.load(pth_2, map_location="cpu")

        if ckpt1["sr"] != ckpt2["sr"]: 
            gr_warning(translations["sr_not_same"])
            return [translations["sr_not_same"], None]

        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        vocoder = ckpt1.get("vocoder", "Default")

        ckpt1 = extract(ckpt1) if "model" in ckpt1 else ckpt1["weight"]
        ckpt2 = extract(ckpt2) if "model" in ckpt2 else ckpt2["weight"]

        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())): 
            gr_warning(translations["architectures_not_same"])
            return [translations["architectures_not_same"], None]
         
        gr_info(translations["start"].format(start=translations["fushion_model"]))

        opt = OrderedDict()
        opt["weight"] = {}

        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (ratio * (ckpt1[key][:min_shape0].float()) + (1 - ratio) * (ckpt2[key][:min_shape0].float())).half()
            else: opt["weight"][key] = (ratio * (ckpt1[key].float()) + (1 - ratio) * (ckpt2[key].float())).half()

        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["infos"] = translations["model_fushion_info"].format(name=name, pth_1=pth_1, pth_2=pth_2, ratio=ratio)
        opt["vocoder"] = vocoder

        output_model = os.path.join("assets", "weights")
        if not os.path.exists(output_model): os.makedirs(output_model, exist_ok=True)

        torch.save(opt, os.path.join(output_model, name))

        gr_info(translations["success"])
        return [translations["success"], os.path.join(output_model, name)]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))
        logger.debug(e)
        return [e, None]

def fushion_model(name, path_1, path_2, ratio):
    if not name:
        gr_warning(translations["provide_name_is_save"]) 
        return [translations["provide_name_is_save"], None]

    if path_1.endswith(".pth") and path_2.endswith(".pth"): return fushion_model_pth(name.replace(".onnx", ".pth"), path_1, path_2, ratio)
    else:
        gr_warning(translations["format_not_valid"])
        return [None, None]
    
def onnx_export(model_path):
    from main.library.algorithm.onnx_export import onnx_exporter
    
    if not model_path.endswith(".pth"): model_path + ".pth"
    if not model_path or not os.path.exists(model_path) or not model_path.endswith(".pth"):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return [None, translations["provide_file"].format(filename=translations["model"])]
    
    try:
        gr_info(translations["start_onnx_export"])
        output = onnx_exporter(model_path, model_path.replace(".pth", ".onnx"), is_half=config.is_half, device=config.device)

        gr_info(translations["success"])
        return [output, translations["success"]]
    except Exception as e:
        return [None, e]
    
def model_info(path):
    if not path or not os.path.exists(path) or os.path.isdir(path) or not path.endswith((".pth", ".onnx")): return gr_warning(translations["provide_file"].format(filename=translations["model"]))
    
    def prettify_date(date_str):
        if date_str == translations["not_found_create_time"]: return None

        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.debug(e)
            return translations["format_not_valid"]
    
    if path.endswith(".pth"): model_data = torch.load(path, map_location=torch.device("cpu"))
    else:
        import onnx
        
        model = onnx.load(path)
        model_data = None

        for prop in model.metadata_props:
            if prop.key == "model_info":
                model_data = json.loads(prop.value)
                break

    gr_info(translations["read_info"])

    epochs = model_data.get("epoch", None)
    if epochs is None: 
        epochs = model_data.get("info", None)
        try:
            epoch = epochs.replace("epoch", "").replace("e", "").isdigit()
            if epoch and epochs is None: epochs = translations["not_found"].format(name=translations["epoch"])
        except: 
            pass

    steps = model_data.get("step", translations["not_found"].format(name=translations["step"]))
    sr = model_data.get("sr", translations["not_found"].format(name=translations["sr"]))
    f0 = model_data.get("f0", translations["not_found"].format(name=translations["f0"]))
    version = model_data.get("version", translations["not_found"].format(name=translations["version"]))
    creation_date = model_data.get("creation_date", translations["not_found_create_time"])
    model_hash = model_data.get("model_hash", translations["not_found"].format(name="model_hash"))
    pitch_guidance = translations["trained_f0"] if f0 else translations["not_f0"]
    creation_date_str = prettify_date(creation_date) if creation_date else translations["not_found_create_time"]
    model_name = model_data.get("model_name", translations["unregistered"])
    model_author = model_data.get("author", translations["not_author"])
    vocoder = model_data.get("vocoder", "Default")

    gr_info(translations["success"])
    return translations["model_info"].format(model_name=model_name, model_author=model_author, epochs=epochs, steps=steps, version=version, sr=sr, pitch_guidance=pitch_guidance, model_hash=model_hash, creation_date_str=creation_date_str, vocoder=vocoder)

def audio_effects(input_path, output_path, resample, resample_sr, chorus_depth, chorus_rate, chorus_mix, chorus_delay, chorus_feedback, distortion_drive, reverb_room_size, reverb_damping, reverb_wet_level, reverb_dry_level, reverb_width, reverb_freeze_mode, pitch_shift, delay_seconds, delay_feedback, delay_mix, compressor_threshold, compressor_ratio, compressor_attack_ms, compressor_release_ms, limiter_threshold, limiter_release, gain_db, bitcrush_bit_depth, clipping_threshold, phaser_rate_hz, phaser_depth, phaser_centre_frequency_hz, phaser_feedback, phaser_mix, bass_boost_db, bass_boost_frequency, treble_boost_db, treble_boost_frequency, fade_in_duration, fade_out_duration, export_format, chorus, distortion, reverb, delay, compressor, limiter, gain, bitcrush, clipping, phaser, treble_bass_boost, fade_in_out, audio_combination, audio_combination_input):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output_path): output_path = os.path.join(output_path, f"audio_effects.{export_format}")
    output_dir = os.path.dirname(output_path) or output_path

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path): os.remove(output_path)
    
    gr_info(translations["start"].format(start=translations["apply_effect"]))
    subprocess.run([python, "main/inference/audio_effects.py", "--input_path", input_path, "--output_path", output_path, "--resample", str(resample), "--resample_sr", str(resample_sr), "--chorus_depth", str(chorus_depth), "--chorus_rate", str(chorus_rate), "--chorus_mix", str(chorus_mix), "--chorus_delay", str(chorus_delay), "--chorus_feedback", str(chorus_feedback), "--drive_db", str(distortion_drive), "--reverb_room_size", str(reverb_room_size), "--reverb_damping", str(reverb_damping), "--reverb_wet_level", str(reverb_wet_level), "--reverb_dry_level", str(reverb_dry_level), "--reverb_width", str(reverb_width), "--reverb_freeze_mode", str(reverb_freeze_mode), "--pitch_shift", str(pitch_shift), "--delay_seconds", str(delay_seconds), "--delay_feedback", str(delay_feedback), "--delay_mix", str(delay_mix), "--compressor_threshold", str(compressor_threshold), "--compressor_ratio", str(compressor_ratio), "--compressor_attack_ms", str(compressor_attack_ms), "--compressor_release_ms", str(compressor_release_ms), "--limiter_threshold", str(limiter_threshold), "--limiter_release", str(limiter_release), "--gain_db", str(gain_db), "--bitcrush_bit_depth", str(bitcrush_bit_depth), "--clipping_threshold", str(clipping_threshold), "--phaser_rate_hz", str(phaser_rate_hz), "--phaser_depth", str(phaser_depth), "--phaser_centre_frequency_hz", str(phaser_centre_frequency_hz), "--phaser_feedback", str(phaser_feedback), "--phaser_mix", str(phaser_mix), "--bass_boost_db", str(bass_boost_db), "--bass_boost_frequency", str(bass_boost_frequency), "--treble_boost_db", str(treble_boost_db), "--treble_boost_frequency", str(treble_boost_frequency), "--fade_in_duration", str(fade_in_duration), "--fade_out_duration", str(fade_out_duration), "--export_format", export_format, "--chorus", str(chorus), "--distortion", str(distortion), "--reverb", str(reverb), "--pitchshift", str(pitch_shift != 0), "--delay", str(delay), "--compressor", str(compressor), "--limiter", str(limiter), "--gain", str(gain), "--bitcrush", str(bitcrush), "--clipping", str(clipping), "--phaser", str(phaser), "--treble_bass_boost", str(treble_bass_boost), "--fade_in_out", str(fade_in_out), "--audio_combination", str(audio_combination), "--audio_combination_input", audio_combination_input])

    gr_info(translations["success"])
    return output_path.replace("wav", export_format)

def synthesize_tts(prompt, voice, speed, output, pitch, google):
    if not google: 
        from edge_tts import Communicate

        asyncio.run(Communicate(text=prompt, voice=voice, rate=f"+{speed}%" if speed >= 0 else f"{speed}%", pitch=f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz").save(output))
    else: 
        response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyr.pbz/genafyngr_ggf", "rot13"), params={"ie": "UTF-8", "q": prompt, "tl": voice, "ttsspeed": speed, "client": "tw-ob"}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"})

        if response.status_code == 200:
            with open(output, "wb") as f:
                f.write(response.content)

            if pitch != 0 or speed != 0:
                y, sr = librosa.load(output, sr=None)

                if pitch != 0: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
                if speed != 0: y = librosa.effects.time_stretch(y, rate=speed)

                sf.write(file=output, data=y, samplerate=sr, format=os.path.splitext(os.path.basename(output))[-1].lower().replace('.', ''))
        else: gr_error(f"{response.status_code}, {response.text}")

def time_stretch(y, sr, target_duration):
    rate = (len(y) / sr) / target_duration
    if rate != 1.0: y = librosa.effects.time_stretch(y=y.astype(np.float32), rate=rate)

    n_target = int(round(target_duration * sr))
    return np.pad(y, (0, n_target - len(y))) if len(y) < n_target else y[:n_target]

def pysrttime_to_seconds(t):
    return (t.hours * 60 + t.minutes) * 60 + t.seconds + t.milliseconds / 1000

def srt_tts(srt_file, out_file, voice, rate = 0, sr = 24000, google = False):
    import pysrt
    import tempfile

    subs = pysrt.open(srt_file)
    if not subs: raise ValueError(translations["srt"])

    final_audio = np.zeros(int(round(pysrttime_to_seconds(subs[-1].end) * sr)), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tempdir:
        for idx, seg in enumerate(subs):
            wav_path = os.path.join(tempdir, f"seg_{idx}.wav")
            synthesize_tts(" ".join(seg.text.splitlines()), voice, 0, wav_path, rate, google)

            audio, file_sr = sf.read(wav_path, dtype=np.float32)
            if file_sr != sr: audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / file_sr)), np.arange(len(audio)), audio)
            adjusted = time_stretch(audio, sr, pysrttime_to_seconds(seg.duration))

            start_sample = int(round(pysrttime_to_seconds(seg.start) * sr))
            end_sample = start_sample + adjusted.shape[0]

            if end_sample > final_audio.shape[0]:
                adjusted = adjusted[: final_audio.shape[0] - start_sample]
                end_sample = final_audio.shape[0]

            final_audio[start_sample:end_sample] += adjusted

    sf.write(out_file, final_audio, sr)

def TTS(prompt, voice, speed, output, pitch, google, srt_input):
    if not srt_input: srt_input = ""

    if not prompt and not srt_input.endswith(".srt"):
        gr_warning(translations["enter_the_text"])
        return None
    
    if not voice:
        gr_warning(translations["choose_voice"])
        return None
    
    if not output: 
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.isdir(output): output = os.path.join(output, f"tts.wav")
    gr_info(translations["convert"].format(name=translations["text"]))

    output_dir = os.path.dirname(output) or output
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    if srt_input.endswith(".srt"): srt_tts(srt_input, output, voice, 0, 24000, google)
    else: synthesize_tts(prompt, voice, speed, output, pitch, google)

    gr_info(translations["success"])
    return output

def separator_music(input, output_audio, format, shifts, segments_size, overlap, clean_audio, clean_strength, denoise, separator_model, kara_model, backing, reverb, backing_reverb, hop_length, batch_size, sample_rate):
    output = os.path.dirname(output_audio) or output_audio

    if not input or not os.path.exists(input) or os.path.isdir(input): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output): 
        gr_warning(translations["output_not_valid"])
        return [None]*4

    if not os.path.exists(output): os.makedirs(output)
    gr_info(translations["start"].format(start=translations["separator_music"]))

    subprocess.run([python, "main/inference/separator_music.py", "--input_path", input, "--output_path", output, "--format", format, "--shifts", str(shifts), "--segments_size", str(segments_size), "--overlap", str(overlap), "--mdx_hop_length", str(hop_length), "--mdx_batch_size", str(batch_size), "--clean_audio", str(clean_audio), "--clean_strength", str(clean_strength), "--kara_model", kara_model, "--backing", str(backing), "--mdx_denoise", str(denoise), "--reverb", str(reverb), "--backing_reverb", str(backing_reverb), "--model_name", separator_model, "--sample_rate", str(sample_rate)])
    gr_info(translations["success"])

    filename, _ = os.path.splitext(os.path.basename(input))
    output = os.path.join(output, filename)

    return [os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}"), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}") if backing else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}") if backing else None)] if os.path.isfile(input) else [None]*4

def convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file):    
    subprocess.run([python, "main/inference/convert.py", "--pitch", str(pitch), "--filter_radius", str(filter_radius), "--index_rate", str(index_rate), "--volume_envelope", str(volume_envelope), "--protect", str(protect), "--hop_length", str(hop_length), "--f0_method", f0_method, "--input_path", input_path, "--output_path", output_path, "--pth_path", pth_path, "--index_path", index_path if index_path else "", "--f0_autotune", str(f0_autotune), "--clean_audio", str(clean_audio), "--clean_strength", str(clean_strength), "--export_format", export_format, "--embedder_model", embedder_model, "--resample_sr", str(resample_sr), "--split_audio", str(split_audio), "--f0_autotune_strength", str(f0_autotune_strength), "--checkpointing", str(checkpointing), "--f0_onnx", str(onnx_f0_mode), "--embedders_mode", embedders_mode, "--formant_shifting", str(formant_shifting), "--formant_qfrency", str(formant_qfrency), "--formant_timbre", str(formant_timbre), "--f0_file", f0_file])

def convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, input_audio_name, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    model_path = os.path.join("assets", "weights", model)

    return_none = [None]*6
    return_none[5] = {"visible": True, "__type__": "update"}

    if not use_audio:
        if merge_instrument or not_merge_backing or convert_backing or use_original:
            gr_warning(translations["turn_on_use_audio"])
            return return_none

    if use_original:
        if convert_backing:
            gr_warning(translations["turn_off_convert_backup"])
            return return_none
        elif not_merge_backing:
            gr_warning(translations["turn_off_merge_backup"])
            return return_none

    if not model or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return return_none

    f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

    if use_audio:
        output_audio = os.path.join("audios", input_audio_name)

        from main.library.utils import pydub_convert, pydub_load
        
        def get_audio_file(label):
            matching_files = [f for f in os.listdir(output_audio) if label in f]

            if not matching_files: return translations["notfound"]   
            return os.path.join(output_audio, matching_files[0])

        output_path = os.path.join(output_audio, f"Convert_Vocals.{format}")
        output_backing = os.path.join(output_audio, f"Convert_Backing.{format}")
        output_merge_backup = os.path.join(output_audio, f"Vocals+Backing.{format}")
        output_merge_instrument = os.path.join(output_audio, f"Vocals+Instruments.{format}")

        if os.path.exists(output_audio): os.makedirs(output_audio, exist_ok=True)
        if os.path.exists(output_path): os.remove(output_path)

        if use_original:
            original_vocal = get_audio_file('Original_Vocals_No_Reverb.')

            if original_vocal == translations["notfound"]: original_vocal = get_audio_file('Original_Vocals.')

            if original_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_original_vocal"])
                return return_none
            
            input_path = original_vocal
        else:
            main_vocal = get_audio_file('Main_Vocals_No_Reverb.')
            backing_vocal = get_audio_file('Backing_Vocals_No_Reverb.')

            if main_vocal == translations["notfound"]: main_vocal = get_audio_file('Main_Vocals.')
            if not not_merge_backing and backing_vocal == translations["notfound"]: backing_vocal = get_audio_file('Backing_Vocals.')

            if main_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_main_vocal"])
                return return_none
            
            if not not_merge_backing and backing_vocal == translations["notfound"]: 
                gr_warning(translations["not_found_backing_vocal"])
                return return_none
            
            input_path = main_vocal
            backing_path = backing_vocal

        gr_info(translations["convert_vocal"])

        convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input_path, output_path, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

        gr_info(translations["convert_success"])

        if convert_backing:
            if os.path.exists(output_backing): os.remove(output_backing)

            gr_info(translations["convert_backup"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, backing_path, output_backing, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["convert_backup_success"])

        try:
            if not not_merge_backing and not use_original:
                backing_source = output_backing if convert_backing else backing_vocal

                if os.path.exists(output_merge_backup): os.remove(output_merge_backup)

                gr_info(translations["merge_backup"])

                pydub_convert(pydub_load(output_path)).overlay(pydub_convert(pydub_load(backing_source))).export(output_merge_backup, format=format)

                gr_info(translations["merge_success"])

            if merge_instrument:    
                vocals = output_merge_backup if not not_merge_backing and not use_original else output_path

                if os.path.exists(output_merge_instrument): os.remove(output_merge_instrument)

                gr_info(translations["merge_instruments_process"])

                instruments = get_audio_file('Instruments.')
                
                if instruments == translations["notfound"]: 
                    gr_warning(translations["not_found_instruments"])
                    output_merge_instrument = None
                else: pydub_convert(pydub_load(instruments)).overlay(pydub_convert(pydub_load(vocals))).export(output_merge_instrument, format=format)
                
                gr_info(translations["merge_success"])
        except:
            return return_none

        return [(None if use_original else output_path), output_backing, (None if not_merge_backing and use_original else output_merge_backup), (output_path if use_original else None), (output_merge_instrument if merge_instrument else None), {"visible": True, "__type__": "update"}]
    else:
        if not input or not os.path.exists(input) or os.path.isdir(input): 
            gr_warning(translations["input_not_valid"])
            return return_none
        
        if not output:
            gr_warning(translations["output_not_valid"])
            return return_none
        
        output = output.replace("wav", format)

        if os.path.isdir(input):
            gr_info(translations["is_folder"])

            if not [f for f in os.listdir(input) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]:
                gr_warning(translations["not_found_in_folder"])
                return return_none
            
            gr_info(translations["batch_convert"])

            output_dir = os.path.dirname(output) or output
            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output_dir, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["batch_convert_success"])

            return return_none
        else:
            output_dir = os.path.dirname(output) or output

            if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output): os.remove(output)

            gr_info(translations["convert_vocal"])

            convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

            gr_info(translations["convert_success"])

            return_none[0] = output
            return return_none

def convert_selection(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    if use_audio:
        gr_info(translations["search_separate"])

        choice = [f for f in os.listdir("audios") if os.path.isdir(os.path.join("audios", f))]

        gr_info(translations["found_choice"].format(choice=len(choice)))

        if len(choice) == 0: 
            gr_warning(translations["separator==0"])

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, None, None, None, None, None, {"visible": True, "__type__": "update"}]
        elif len(choice) == 1:
            convert_output = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, None, None, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, choice[0], checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode)

            return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, convert_output[0], convert_output[1], convert_output[2], convert_output[3], convert_output[4], {"visible": True, "__type__": "update"}]
        else: return [{"choices": choice, "value": "", "interactive": True, "visible": True, "__type__": "update"}, None, None, None, None, None, {"visible": False, "__type__": "update"}]
    else:
        main_convert = convert_audio(clean, autotune, use_audio, use_original, convert_backing, not_merge_backing, merge_instrument, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, None, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode)

        return [{"choices": [], "value": "", "interactive": False, "visible": False, "__type__": "update"}, main_convert[0], None, None, None, None, {"visible": True, "__type__": "update"}]
    
def convert_with_whisper(num_spk, model_size, cleaner, clean_strength, autotune, f0_autotune_strength, checkpointing, model_1, model_2, model_index_1, model_index_2, pitch_1, pitch_2, index_strength_1, index_strength_2, export_format, input_audio, output_audio, onnx_f0_mode, method, hybrid_method, hop_length, embed_mode, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, formant_shifting, formant_qfrency_1, formant_timbre_1, formant_qfrency_2, formant_timbre_2):
    from pydub import AudioSegment
    from sklearn.cluster import AgglomerativeClustering
    
    from main.library.speaker_diarization.audio import Audio
    from main.library.speaker_diarization.segment import Segment
    from main.library.speaker_diarization.whisper import load_model
    from main.library.utils import check_spk_diarization, pydub_convert, pydub_load
    from main.library.speaker_diarization.embedding import SpeechBrainPretrainedSpeakerEmbedding
    
    check_spk_diarization(model_size)
    model_pth_1, model_pth_2 = os.path.join("assets", "weights", model_1), os.path.join("assets", "weights", model_2)

    if (not model_1 or not os.path.exists(model_pth_1) or os.path.isdir(model_pth_1) or not model_pth_1.endswith((".pth", ".onnx"))) and (not model_2 or not os.path.exists(model_pth_2) or os.path.isdir(model_pth_2) or not model_pth_2.endswith((".pth", ".onnx"))):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None
    
    if not model_1: model_pth_1 = model_pth_2
    if not model_2: model_pth_2 = model_pth_1

    if not input_audio or not os.path.exists(input_audio) or os.path.isdir(input_audio): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_audio:
        gr_warning(translations["output_not_valid"])
        return None
    
    if os.path.exists(output_audio): os.remove(output_audio)
    gr_info(translations["start_whisper"])
    
    try:
        audio = Audio()

        embedding_model = SpeechBrainPretrainedSpeakerEmbedding(device=config.device)
        segments = load_model(model_size, device=config.device).transcribe(input_audio, fp16=configs.get("fp16", False), word_timestamps=True)["segments"]

        y, sr = librosa.load(input_audio, sr=None)  
        duration = len(y) / sr
            
        def segment_embedding(segment):
            waveform, _ = audio.crop(input_audio, Segment(segment["start"], min(duration, segment["end"])))
            return embedding_model(waveform.mean(dim=0, keepdim=True)[None] if waveform.shape[0] == 2 else waveform[None])  
        
        def time(secs):
            return datetime.timedelta(seconds=round(secs))
        
        def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
            def extract_number(filename):
                match = re.search(r'_(\d+)', filename)
                return int(match.group(1)) if match else 0

            total_duration = len(pydub_load(original_file_path))
            combined = AudioSegment.empty() 
            current_position = 0 

            for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
                if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
                
                combined += pydub_load(file)  
                current_position = end_i

            if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
            combined.export(output_path, format=format)

            return output_path

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        labels = AgglomerativeClustering(num_spk).fit(np.nan_to_num(embeddings)).labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        merged_segments, current_text = [], []
        current_speaker, current_start = None, None

        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            start_time = segment["start"]
            text = segment["text"][1:]  

            if speaker == current_speaker:
                current_text.append(text)
                end_time = segment["end"]
            else:
                if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})
                
                current_speaker = speaker
                current_start = start_time
                current_text = [text]
                end_time = segment["end"]

        if current_speaker is not None: merged_segments.append({"speaker": current_speaker, "start": current_start, "end": end_time, "text": " ".join(current_text)})

        gr_info(translations["whisper_done"])

        x = ""
        for segment in merged_segments:
            x += f"\n{segment['speaker']} {str(time(segment['start']))} - {str(time(segment['end']))}\n"
            x += segment["text"] + "\n"

        logger.info(x)

        gr_info(translations["process_audio"])

        audio = pydub_convert(pydub_load(input_audio))
        output_folder = "audios_temp"

        if os.path.exists(output_folder): shutil.rmtree(output_folder, ignore_errors=True)
        for f in [output_folder, os.path.join(output_folder, "1"), os.path.join(output_folder, "2")]:
            os.makedirs(f, exist_ok=True)

        time_stamps, processed_segments = [], []
        for i, segment in enumerate(merged_segments):
            start_ms = int(segment["start"] * 1000) 
            end_ms = int(segment["end"] * 1000)

            index = i + 1

            segment_filename = os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}.wav")
            audio[start_ms:end_ms].export(segment_filename, format="wav")

            processed_segments.append(os.path.join(output_folder, "1" if i % 2 == 1 else "2", f"segment_{index}_output.wav"))
            time_stamps.append((start_ms, end_ms))

        f0method, embedder_model = (method if method != "hybrid" else hybrid_method), (embedders if embedders != "custom" else custom_embedders)

        gr_info(translations["process_done_start_convert"])

        convert(pitch_1, filter_radius, index_strength_1, volume_envelope, protect, hop_length, f0method, os.path.join(output_folder, "1"), output_folder, model_pth_1, model_index_1, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_1, formant_timbre_1, "")
        convert(pitch_2, filter_radius, index_strength_2, volume_envelope, protect, hop_length, f0method, os.path.join(output_folder, "2"), output_folder, model_pth_2, model_index_2, autotune, cleaner, clean_strength, "wav", embedder_model, resample_sr, False, f0_autotune_strength, checkpointing, onnx_f0_mode, embed_mode, formant_shifting, formant_qfrency_2, formant_timbre_2, "")

        gr_info(translations["convert_success"])
        return merge_audio(processed_segments, time_stamps, input_audio, output_audio.replace("wav", export_format), export_format)
    except Exception as e:
        gr_error(translations["error_occurred"].format(e=e))
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        if os.path.exists("audios_temp"): shutil.rmtree("audios_temp", ignore_errors=True)

def convert_tts(clean, autotune, pitch, clean_strength, model, index, index_rate, input, output, format, method, hybrid_method, hop_length, embedders, custom_embedders, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file, embedders_mode):
    model_path = os.path.join("assets", "weights", model)

    if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path) or not model.endswith((".pth", ".onnx")):
        gr_warning(translations["provide_file"].format(filename=translations["model"]))
        return None

    if not input or not os.path.exists(input): 
        gr_warning(translations["input_not_valid"])
        return None
    
    if os.path.isdir(input): 
        input_audio = [f for f in os.listdir(input) if "tts" in f and f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]
        
        if not input_audio:
            gr_warning(translations["not_found_in_folder"])
            return None
        
        input = os.path.join(input, input_audio[0])
    
    if not output:
        gr_warning(translations["output_not_valid"])
        return None
    
    output = output.replace("wav", format)
    if os.path.isdir(output): output = os.path.join(output, f"tts.{format}")

    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output): os.remove(output)

    f0method = method if method != "hybrid" else hybrid_method
    embedder_model = embedders if embedders != "custom" else custom_embedders

    gr_info(translations["convert_vocal"])

    convert(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0method, input, output, model_path, index, autotune, clean, clean_strength, format, embedder_model, resample_sr, split_audio, f0_autotune_strength, checkpointing, onnx_f0_mode, embedders_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file)

    gr_info(translations["convert_success"])
    return output

def log_read(log_file, done):
    f = open(log_file, "w", encoding="utf-8")
    f.close()

    while 1:
        with open(log_file, "r", encoding="utf-8") as f:
            yield "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

        sleep(1)
        if done[0]: break

    with open(log_file, "r", encoding="utf-8") as f:
        log = "".join(line for line in f.readlines() if "DEBUG" not in line and line.strip() != "")

    yield log

def create_dataset(input_audio, output_dataset, clean_dataset, clean_strength, separator_reverb, kim_vocals_version, overlap, segments_size, denoise_mdx, skip, skip_start, skip_end, hop_length, batch_size, sample_rate):
    version = 1 if kim_vocals_version == "Version-1" else 2

    gr_info(translations["start"].format(start=translations["create"]))

    p = subprocess.Popen(f'{python} main/inference/create_dataset.py --input_audio "{input_audio}" --output_dataset "{output_dataset}" --clean_dataset {clean_dataset} --clean_strength {clean_strength} --separator_reverb {separator_reverb} --kim_vocal_version {version} --overlap {overlap} --segments_size {segments_size} --mdx_hop_length {hop_length} --mdx_batch_size {batch_size} --denoise_mdx {denoise_mdx} --skip {skip} --skip_start_audios "{skip_start}" --skip_end_audios "{skip_end}" --sample_rate {sample_rate}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(os.path.join("assets", "logs", "create_dataset.log"), done):
        yield log

def preprocess(model_name, sample_rate, cpu_core, cut_preprocess, process_effects, path, clean_dataset, clean_strength):
    dataset = os.path.join(path)
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])
    if not any(f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3")) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))): return gr_warning(translations["not_found_data"])
    
    model_dir = os.path.join("assets", "logs", model_name)
    if os.path.exists(model_dir): shutil.rmtree(model_dir, ignore_errors=True)

    p = subprocess.Popen(f'{python} main/inference/preprocess.py --model_name "{model_name}" --dataset_path "{dataset}" --sample_rate {sr} --cpu_cores {cpu_core} --cut_preprocess {cut_preprocess} --process_effects {process_effects} --clean_dataset {clean_dataset} --clean_strength {clean_strength}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "preprocess.log"), done):
        yield log

def extract(model_name, version, method, pitch_guidance, hop_length, cpu_cores, gpu, sample_rate, embedders, custom_embedders, onnx_f0_mode, embedders_mode):
    embedder_model = embedders if embedders != "custom" else custom_embedders
    sr = int(float(sample_rate.rstrip("k")) * 1000)

    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join("assets", "logs", model_name)
    if not any(os.path.isfile(os.path.join(model_dir, "sliced_audios", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios"))) or not any(os.path.isfile(os.path.join(model_dir, "sliced_audios_16k", f)) for f in os.listdir(os.path.join(model_dir, "sliced_audios_16k"))): return gr_warning(translations["not_found_data_preprocess"])

    p = subprocess.Popen(f'{python} main/inference/extract.py --model_name "{model_name}" --rvc_version {version} --f0_method {method} --pitch_guidance {pitch_guidance} --hop_length {hop_length} --cpu_cores {cpu_cores} --gpu {gpu} --sample_rate {sr} --embedder_model {embedder_model} --f0_onnx {onnx_f0_mode} --embedders_mode {embedders_mode}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "extract.log"), done):
        yield log

def create_index(model_name, rvc_version, index_algorithm):
    if not model_name: return gr_warning(translations["provide_name"])
    model_dir = os.path.join("assets", "logs", model_name)

    if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])

    p = subprocess.Popen(f'{python} main/inference/create_index.py --model_name "{model_name}" --rvc_version {rvc_version} --index_algorithm {index_algorithm}', shell=True)
    done = [False]

    threading.Thread(target=if_done, args=(done, p)).start()
    os.makedirs(model_dir, exist_ok=True)

    for log in log_read(os.path.join(model_dir, "create_index.log"), done):
        yield log

def training(model_name, rvc_version, save_every_epoch, save_only_latest, save_every_weights, total_epoch, sample_rate, batch_size, gpu, pitch_guidance, not_pretrain, custom_pretrained, pretrain_g, pretrain_d, detector, threshold, clean_up, cache, model_author, vocoder, checkpointing, deterministic, benchmark):
    sr = int(float(sample_rate.rstrip("k")) * 1000)
    if not model_name: return gr_warning(translations["provide_name"])

    model_dir = os.path.join("assets", "logs", model_name)
    if os.path.exists(os.path.join(model_dir, "train_pid.txt")): os.remove(os.path.join(model_dir, "train_pid.txt"))

    if not any(os.path.isfile(os.path.join(model_dir, f"{rvc_version}_extracted", f)) for f in os.listdir(os.path.join(model_dir, f"{rvc_version}_extracted"))): return gr_warning(translations["not_found_data_extract"])

    if not not_pretrain:
        if not custom_pretrained: 
            pretrained_selector = {True: {32000: ("f0G32k.pth", "f0D32k.pth"), 40000: ("f0G40k.pth", "f0D40k.pth"), 48000: ("f0G48k.pth", "f0D48k.pth")}, False: {32000: ("G32k.pth", "D32k.pth"), 40000: ("G40k.pth", "D40k.pth"), 48000: ("G48k.pth", "D48k.pth")}}

            pg, pd = pretrained_selector[pitch_guidance][sr]
        else:
            if not pretrain_g: return gr_warning(translations["provide_pretrained"].format(dg="G"))
            if not pretrain_d: return gr_warning(translations["provide_pretrained"].format(dg="D"))
            
            pg, pd = pretrain_g, pretrain_d

        pretrained_G, pretrained_D = (os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pg}" if vocoder != 'Default' else pg), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pd}" if vocoder != 'Default' else pd)) if not custom_pretrained else (os.path.join("assets", "models", f"pretrained_custom", pg), os.path.join("assets", "models", f"pretrained_custom", pd))
        download_version = codecs.decode(f"uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_i{'2' if rvc_version == 'v2' else '1'}/", "rot13")
        
        if not custom_pretrained:
            try:
                if not os.path.exists(pretrained_G):
                    gr_info(translations["download_pretrained"].format(dg="G", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pg]) if vocoder != 'Default' else (download_version + pg), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pg}" if vocoder != 'Default' else pg))
                        
                if not os.path.exists(pretrained_D):
                    gr_info(translations["download_pretrained"].format(dg="D", rvc_version=rvc_version))
                    huggingface.HF_download_file("".join([download_version, vocoder, "_", pd]) if vocoder != 'Default' else (download_version + pd), os.path.join("assets", "models", f"pretrained_{rvc_version}", f"{vocoder}_{pd}" if vocoder != 'Default' else pd))
            except:
                gr_warning(translations["not_use_pretrain_error_download"])
                pretrained_G, pretrained_D = None, None
        else:
            if not os.path.exists(pretrained_G): return gr_warning(translations["not_found_pretrain"].format(dg="G"))
            if not os.path.exists(pretrained_D): return gr_warning(translations["not_found_pretrain"].format(dg="D"))
    else: gr_warning(translations["not_use_pretrain"])

    gr_info(translations["start"].format(start=translations["training"]))

    p = subprocess.Popen(f'{python} main/inference/train.py --model_name "{model_name}" --rvc_version {rvc_version} --save_every_epoch {save_every_epoch} --save_only_latest {save_only_latest} --save_every_weights {save_every_weights} --total_epoch {total_epoch} --sample_rate {sr} --batch_size {batch_size} --gpu {gpu} --pitch_guidance {pitch_guidance} --overtraining_detector {detector} --overtraining_threshold {threshold} --cleanup {clean_up} --cache_data_in_gpu {cache} --g_pretrained_path "{pretrained_G}" --d_pretrained_path "{pretrained_D}" --model_author "{model_author}" --vocoder "{vocoder}" --checkpointing {checkpointing} --deterministic {deterministic} --benchmark {benchmark}', shell=True)
    done = [False]

    with open(os.path.join(model_dir, "train_pid.txt"), "w") as pid_file:
        pid_file.write(str(p.pid))

    threading.Thread(target=if_done, args=(done, p)).start()

    for log in log_read(os.path.join(model_dir, "train.log"), done):
        if len(log.split("\n")) > 100: log = log[-100:]
        yield log

def stop_pid(pid_file, model_name=None, train=False):
    try:
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join("assets", "logs", model_name, f"{pid_file}.txt")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                os.kill(pid, 9)

            if os.path.exists(pid_file_path): os.remove(pid_file_path)

        pid_file_path = os.path.join("assets", "logs", model_name, "config.json")

        if train and os.path.exists(pid_file_path):
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])
    except:
        pass

def load_presets(presets, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, formant_shifting, formant_qfrency, formant_timbre):
    if not presets: return gr_warning(translations["provide_file_settings"])

    with open(os.path.join("assets", "presets", presets)) as f:
        file = json.load(f)

    gr_info(translations["load_presets"].format(presets=presets))
    return file.get("cleaner", cleaner), file.get("autotune", autotune), file.get("pitch", pitch), file.get("clean_strength", clean_strength), file.get("index_strength", index_strength), file.get("resample_sr", resample_sr), file.get("filter_radius", filter_radius), file.get("volume_envelope", volume_envelope), file.get("protect", protect), file.get("split_audio", split_audio), file.get("f0_autotune_strength", f0_autotune_strength), file.get("formant_shifting", formant_shifting), file.get("formant_qfrency", formant_qfrency), file.get("formant_timbre", formant_timbre)

def save_presets(name, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox, formant_shifting, formant_qfrency, formant_timbre):  
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox]): return gr_warning(translations["choose1"])

    settings = {}

    for checkbox, data in [(cleaner_chbox, {"cleaner": cleaner, "clean_strength": clean_strength}), (autotune_chbox, {"autotune": autotune, "f0_autotune_strength": f0_autotune_strength}), (pitch_chbox, {"pitch": pitch}), (index_strength_chbox, {"index_strength": index_strength}), (resample_sr_chbox, {"resample_sr": resample_sr}), (filter_radius_chbox, {"filter_radius": filter_radius}), (volume_envelope_chbox, {"volume_envelope": volume_envelope}), (protect_chbox, {"protect": protect}), (split_audio_chbox, {"split_audio": split_audio}), (formant_shifting_chbox, {"formant_shifting": formant_shifting, "formant_qfrency": formant_qfrency, "formant_timbre": formant_timbre})]:
        if checkbox: settings.update(data)

    with open(os.path.join("assets", "presets", name + ".json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"])
    return change_preset_choices()

def report_bug(error_info, provide):
    report_path = os.path.join("assets", "logs", "report_bugs.log")
    if os.path.exists(report_path): os.remove(report_path)

    report_url = codecs.decode(requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/jroubbx.gkg", "rot13")).text, "rot13")
    if not error_info: error_info = "Không Có"

    gr_info(translations["thank"])

    if provide:
        try:
            for log in [os.path.join(root, name) for root, _, files in os.walk(os.path.join("assets", "logs"), topdown=False) for name in files if name.endswith(".log")]:
                with open(log, "r", encoding="utf-8") as r:
                    with open(report_path, "a", encoding="utf-8") as w:
                        w.write(str(r.read()))
                        w.write("\n")
        except Exception as e:
            gr_error(translations["error_read_log"])
            logger.debug(e)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": f"Mô tả lỗi: {error_info}", "color": 15158332, "author": {"name": "Vietnamese_RVC", "icon_url": codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/vpb.cat", "rot13"), "url": codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/gerr/znva","rot13")}, "thumbnail": {"url": codecs.decode("uggcf://p.grabe.pbz/7dADJbv-36fNNNNq/grabe.tvs", "rot13")}, "fields": [{"name": "Số Lượng Gỡ Lỗi", "value": content.count("DEBUG")}, {"name": "Số Lượng Thông Tin", "value": content.count("INFO")}, {"name": "Số Lượng Cảnh Báo", "value": content.count("WARNING")}, {"name": "Số Lượng Lỗi", "value": content.count("ERROR")}], "footer": {"text": f"Tên Máy: {platform.uname().node} - Hệ Điều Hành: {platform.system()}-{platform.version()}\nThời Gian Báo Cáo Lỗi: {datetime.datetime.now()}."}}]})

            with open(report_path, "rb") as f:
                requests.post(report_url, files={"file": f})
        except Exception as e:
            gr_error(translations["error_send"])
            logger.debug(e)
        finally:
            if os.path.exists(report_path): os.remove(report_path)
    else: requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": error_info}]})

def f0_extract(audio, f0_method, f0_onnx):
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2

    from matplotlib import pyplot as plt
    from main.library.utils import check_predictors
    from main.inference.extract import FeatureInput

    check_predictors(f0_method, f0_onnx)

    f0_path = os.path.join("assets", "f0", os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])

    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    y, sr = librosa.load(audio, sr=None)

    feats = FeatureInput(sample_rate=sr, is_half=config.is_half, device=config.device)
    feats.f0_max = 1600.0

    F_temp = np.array(feats.compute_f0(y.flatten(), f0_method, 160, f0_onnx), dtype=np.float32)
    F_temp[F_temp == 0] = np.nan

    f0 = 1200 * np.log2(F_temp / librosa.midi_to_hz(0))

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(f0):
            f.write(f"{i * sr / 160},{f0_value}\n")

    gr_info(translations["extract_done"])

    return [txt_path, image_path]

def pitch_guidance_lock(vocoders):
    return {"value": True, "interactive": vocoders == "Default", "__type__": "update"}

def vocoders_lock(pitch, vocoders):
    return {"value": vocoders if pitch else "Default", "interactive": pitch, "__type__": "update"}

def run_audioldm2(input_path, output_path, export_format, sample_rate, audioldm_model, source_prompt, target_prompt, steps, cfg_scale_src, cfg_scale_tar, t_start, save_compute):
    if not input_path or not os.path.exists(input_path) or os.path.isdir(input_path): 
        gr_warning(translations["input_not_valid"])
        return None
        
    if not output_path:
        gr_warning(translations["output_not_valid"])
        return None
    
    output_path = output_path.replace("wav", export_format)

    if os.path.exists(output_path): os.remove(output_path)

    gr_info(translations["start_edit"].format(input_path=input_path))
    subprocess.run([python, "main/inference/audioldm2.py", "--input_path", input_path, "--output_path", output_path, "--export_format", str(export_format), "--sample_rate", str(sample_rate), "--audioldm_model", audioldm_model, "--source_prompt", source_prompt, "--target_prompt", target_prompt, "--steps", str(steps), "--cfg_scale_src", str(cfg_scale_src), "--cfg_scale_tar", str(cfg_scale_tar), "--t_start", str(t_start), "--save_compute", str(save_compute)])
    
    gr_info(translations["success"])
    return output_path

def change_fp(fp):
    fp16 = fp == "fp16"

    if fp16 and config.device == "cpu": 
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

def unlock_f0(value):
    return {"choices": method_f0_full if value else method_f0, "value": "rmvpe", "__type__": "update"} 

def unlock_vocoder(value, vocoder):
    return {"value": vocoder if value == "v2" else "Default", "interactive": value == "v2", "__type__": "update"} 

def unlock_ver(value, vocoder):
    return {"value": "v2" if vocoder == "Default" else value, "interactive": vocoder == "Default", "__type__": "update"}

def visible_embedders(value):
    return {"visible": value != "spin", "__type__": "update"}

