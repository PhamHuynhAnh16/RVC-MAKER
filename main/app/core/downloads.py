import os
import re
import sys
import json
import codecs
import shutil
import yt_dlp
import warnings
import requests

from bs4 import BeautifulSoup

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.app.core.ui import gr_info, gr_warning, gr_error, process_output
from main.app.variables import logger, translations, model_options, configs
from main.app.core.process import move_files_from_directory, fetch_pretrained_data

cookies_dir = os.path.join(os.getcwd(), "assets", "youtube", "config.txt")

def download_url(url):
    if not url: return gr_warning(translations["provide_url"])
    if not os.path.exists(configs["audios_path"]): os.makedirs(configs["audios_path"], exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors":
            [{
                "key": "FFmpegExtractAudio", 
                "preferredcodec": "wav", 
                "preferredquality": "192"
            }],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "verbose": False,
            "cookies", cookies_dir
        }

        gr_info(translations["start"].format(start=translations["download_music"]))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_output = os.path.join(configs["audios_path"], re.sub(r'\s+', '-', re.sub(r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', '', ydl.extract_info(url, download=False).get('title', 'video')).strip()))
            if os.path.exists(audio_output): shutil.rmtree(audio_output, ignore_errors=True)

            ydl_opts['outtmpl'] = audio_output
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            audio_output = process_output(audio_output + ".wav")
            
            ydl.download([url])

        gr_info(translations["success"])
        return [audio_output, audio_output, translations["success"]]

def download_model(url=None, model=None):
    if not url: return gr_warning(translations["provide_url"])
    if not model: return gr_warning(translations["provide_name_is_save"])

    model = model.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    download_dir = "download_model"
    weights_dir = configs["weights_path"]
    logs_dir = configs["logs_path"]

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
        return translations["error_occurred"].format(e=e)
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)
        
def download_pretrained_model(choices, model, sample_rate):
    pretraineds_custom_path = configs["pretrained_custom_path"]
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
        
        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag, url_tag = row.find("a", {"class": "fs-5"}), row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                url = url_tag["href"].replace("https://easyaivoice.com/run?url=", "")
                if "huggingface" in url:
                    if name_tag and url_tag: model_options[name_tag.text.replace(".onnx", "").replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "_").replace("-_-", "_").replace("_-_", "_").replace("-", "_").replace("---", "_").replace("___", "_").strip()] = url

        gr_info(translations["found"].format(results=len(model_options)))
        return [{"value": "", "choices": model_options, "interactive": True, "visible": True, "__type__": "update"}, {"value": translations["downloads"], "visible": True, "__type__": "update"}]
