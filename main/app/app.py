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
from main.app.tabs.inference.inference import inference_tabs
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
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            gr.Markdown(f"## {translations['separator_tab']}")
            with gr.Row(): 
                gr.Markdown(translations["4_part"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row(equal_height=True):       
                            cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True, min_width=140)       
                            backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True, min_width=140)
                            reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True, min_width=140)
                            backing_reverb = gr.Checkbox(label=translations["dereveb_backing"], value=False, interactive=False, min_width=140)               
                            denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False, min_width=140)     
                        with gr.Row(equal_height=True):
                            separator_model = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
                            separator_backing_model = gr.Dropdown(label=translations["separator_backing_model"], value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=backing.value)
          
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            shifts = gr.Slider(label=translations["shift"], info=translations["shift_info"], minimum=1, maximum=20, value=2, step=1, interactive=True)
                            segment_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                        with gr.Row():
                            mdx_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():
                            mdx_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True, visible=backing.value or reverb.value or separator_model.value in mdx_model)
                with gr.Column():
                    with gr.Row():
                        clean_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner.value)
                        sample_rate1 = gr.Slider(minimum=8000, maximum=96000, step=1, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
            input = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"]) 
            audio_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
            with gr.Column():
                with gr.Accordion(translations["use_url"], open=False):
                    url = gr.Textbox(label=translations["url_audio"], value="", placeholder="https://www.youtube.com/...", scale=6)
                    download_button = gr.Button(translations["downloads"])
            with gr.Column():
                with gr.Accordion(translations["input_output"], open=False):
                    format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                    input_audio = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                    refesh_separator = gr.Button(translations["refesh"])
                output_separator = gr.Textbox(label=translations["output_folder"], value="audios", placeholder="audios", info=translations["output_folder_info"], interactive=True)
            separator_button = gr.Button(translations["separator_tab"], variant="primary")
            with gr.Row():
                gr.Markdown(translations["output_separator"])
            with gr.Row():
                instruments_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["instruments"])
                original_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["original_vocal"])
                main_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["main_vocal"], visible=backing.value)
                backing_vocals = gr.Audio(show_download_button=True, interactive=False, label=translations["backing_vocal"], visible=backing.value)
            with gr.Row():
                separator_model.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(c not in mdx_model)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, shifts])
                backing.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), visible(a), visible(a), visible(a), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, separator_backing_model, main_vocals, backing_vocals, backing_reverb])
                reverb.change(fn=lambda a, b, c: [visible(a or b or c in mdx_model), visible(a or b or c in mdx_model), valueFalse_interactive(a or b or c in mdx_model), valueFalse_interactive(a and b)], inputs=[backing, reverb, separator_model], outputs=[mdx_batch_size, mdx_hop_length, denoise, backing_reverb])
            with gr.Row():
                input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[audio_input])
                cleaner.change(fn=visible, inputs=[cleaner], outputs=[clean_strength])
            with gr.Row():
                input.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input], outputs=[input_audio])
                refesh_separator.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
            with gr.Row():
                download_button.click(
                    fn=download_url, 
                    inputs=[url], 
                    outputs=[input_audio, audio_input, url],
                    api_name='download_url'
                )
                separator_button.click(
                    fn=separator_music, 
                    inputs=[
                        input_audio, 
                        output_separator,
                        format, 
                        shifts, 
                        segment_size, 
                        overlap, 
                        cleaner, 
                        clean_strength, 
                        denoise, 
                        separator_model, 
                        separator_backing_model, 
                        backing,
                        reverb, 
                        backing_reverb,
                        mdx_hop_length,
                        mdx_batch_size,
                        sample_rate1
                    ],
                    outputs=[original_vocals, instruments_audio, main_vocals, backing_vocals],
                    api_name='separator_music'
                )
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
