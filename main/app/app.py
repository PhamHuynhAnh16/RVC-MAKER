import os
import sys
import torch
import shutil
import librosa
import logging
import requests
import subprocess
import numpy as np
import gradio as gr
import soundfile as sf
from time import sleep
from multiprocessing import cpu_count

sys.path.append(os.getcwd())
from main.app.tabs.inference.inference import inference_tabs, uvr_tabs
from main.app.tabs.models.model import model_tabs
from main.app.tabs.utils.utils import utils_tabs
from main.tools import huggingface
from main.configs.config import Config
from main.app.based.utils import *








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


with gr.Blocks(title="Ultimate RVC Maker ⚡", theme=theme) as app:
    gr.HTML("<h1 style='text-align: center;'>Ultimate RVC Maker ⚡</h1>")
    
    with gr.Tabs():      
        with gr.TabItem("Inference"):
            inference_tabs()
        with gr.TabItem("Model Options"):
            model_tabs()
        uvr_tabs()
        utils_tabs()     
        with gr.TabItem(translations["settings"], visible=configs.get("settings_tab", True)):
            gr.Markdown(translations["settings_markdown"])
            with gr.Row():
                gr.Markdown(translations["settings_markdown_2"])
            with gr.Row():
                toggle_button = gr.Button(translations["change_light_dark"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    language_dropdown = gr.Dropdown(label=translations["lang"], interactive=True, info=translations["lang_restart"], choices=configs.get("support_language", "vi-VN"), value=language)
                    change_lang = gr.Button(translations["change_lang"], variant="primary", scale=2)
                with gr.Column():
                    theme_dropdown = gr.Dropdown(label=translations["theme"], interactive=True, info=translations["theme_restart"], choices=configs.get("themes", theme), value=theme, allow_custom_value=True)
                    changetheme = gr.Button(translations["theme_button"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    fp_choice = gr.Radio(choices=["fp16","fp32"], value="fp16" if configs.get("fp16", False) else "fp32", label=translations["precision"], info=translations["precision_info"], interactive=True)
                    fp_button = gr.Button(translations["update_precision"], variant="secondary", scale=2)
                with gr.Column():
                    font_choice = gr.Textbox(label=translations["font"], info=translations["font_info"], value=font, interactive=True)
                    font_button = gr.Button(translations["change_font"])
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["stop"], open=False):
                        separate_stop = gr.Button(translations["stop_separate"])
                        convert_stop = gr.Button(translations["stop_convert"])
                        create_dataset_stop = gr.Button(translations["stop_create_dataset"])
                        audioldm2_stop = gr.Button(translations["stop_audioldm2"])
                        with gr.Accordion(translations["stop_training"], open=False):
                            model_name_stop = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                            preprocess_stop = gr.Button(translations["stop_preprocess"])
                            extract_stop = gr.Button(translations["stop_extract"])
                            train_stop = gr.Button(translations["stop_training"])
            with gr.Row():
                toggle_button.click(fn=None, js="() => {document.body.classList.toggle('dark')}")
                fp_button.click(fn=change_fp, inputs=[fp_choice], outputs=[fp_choice])
            with gr.Row():
                change_lang.click(fn=change_language, inputs=[language_dropdown], outputs=[])
                changetheme.click(fn=change_theme, inputs=[theme_dropdown], outputs=[])
                font_button.click(fn=change_font, inputs=[font_choice], outputs=[])
            with gr.Row():
                change_lang.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
                changetheme.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
                font_button.click(fn=None, js="setTimeout(function() {location.reload()}, 15000)", inputs=[], outputs=[])
            with gr.Row():
                separate_stop.click(fn=lambda: stop_pid("separate_pid", None, False), inputs=[], outputs=[])
                convert_stop.click(fn=lambda: stop_pid("convert_pid", None, False), inputs=[], outputs=[])
                create_dataset_stop.click(fn=lambda: stop_pid("create_dataset_pid", None, False), inputs=[], outputs=[])
            with gr.Row():
                preprocess_stop.click(fn=lambda model_name_stop: stop_pid("preprocess_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
                extract_stop.click(fn=lambda model_name_stop: stop_pid("extract_pid", model_name_stop, False), inputs=[model_name_stop], outputs=[])
                train_stop.click(fn=lambda model_name_stop: stop_pid("train_pid", model_name_stop, True), inputs=[model_name_stop], outputs=[])
            with gr.Row():
                audioldm2_stop.click(fn=lambda: stop_pid("audioldm2_pid", None, False), inputs=[], outputs=[])
                
    
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])
        gr.Markdown(translations["exemption"])

    logger.info(translations["start_app"])
    logger.info(translations["set_lang"].format(lang=language))

    port = configs.get("app_port", 7860)

    for i in range(configs.get("num_of_restart", 5)):
        try:
            app.queue().launch(
                favicon_path=os.path.join("assets", "ico.png"), 
                server_name=configs.get("server_name", "0.0.0.0"), 
                server_port=port, 
                show_error=configs.get("app_show_error", False), 
                inbrowser="--open" in sys.argv, 
                share="--share" in sys.argv, 
                allowed_paths=allow_disk
            )
            break
        except OSError:
            logger.debug(translations["port"].format(port=port))
            port -= 1
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            sys.exit(1)
