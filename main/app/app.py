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
from main.app.based.utils import *

with gr.Blocks(title=" Ultimate RVC Maker ⚡", theme=theme) as app:
    gr.HTML("<h1 style='text-align: center;'>Ultimate RVC Maker ⚡</h1>")
    
    with gr.Tabs():      
        with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
            gr.Markdown(f"## {translations['separator_tab']}")
            with gr.Row(): 
                gr.Markdown(translations["4_part"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():       
                            cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True, min_width=140)       
                            backing = gr.Checkbox(label=translations["separator_backing"], value=False, interactive=True, min_width=140)
                            reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True, min_width=140)
                            backing_reverb = gr.Checkbox(label=translations["dereveb_backing"], value=False, interactive=False, min_width=140)               
                            denoise = gr.Checkbox(label=translations["denoise_mdx"], value=False, interactive=False, min_width=140)     
                        with gr.Row():
                            separator_model = gr.Dropdown(label=translations["separator_model"], value=uvr_model[0], choices=uvr_model, interactive=True)
                            separator_backing_model = gr.Dropdown(label=translations["separator_backing_model"], value="Version-1", choices=["Version-1", "Version-2"], interactive=True, visible=backing.value)
            with gr.Row():
                with gr.Column():
                    separator_button = gr.Button(translations["separator_tab"], variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
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
                
                with gr.Column():
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

        with gr.TabItem(translations["convert_audio"], visible=configs.get("convert_tab", True)):
            gr.Markdown(f"## {translations['convert_audio']}")
            with gr.Row():
                gr.Markdown(translations["convert_info"])
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"], open=True):
                        with gr.Row():
                            model_pth = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        refesh = gr.Button(translations["refesh"])
                    
                    
            with gr.Row(): 
                with gr.Column():
                    audio_select = gr.Dropdown(label=translations["select_separate"], choices=[], value="", interactive=True, allow_custom_value=True, visible=False)
                    convert_button_2 = gr.Button(translations["convert_audio"], visible=False)
            with gr.Row(): 
                with gr.Column():
                    input0 = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])  
                    play_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
            with gr.Row(): 
                with gr.Column():
                    with gr.Row():
                        index_strength = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index.value != "")
            with gr.Row(): 
                with gr.Column():
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audio0 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audio = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh0 = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Row():
                            cleaner0 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                            checkpointing = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                        with gr.Row():
                            use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=use_audio.value) 
                            convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=use_audio.value)   
                            not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=use_audio.value)
                            merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=use_audio.value) 
                        with gr.Row():
                            pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                            clean_strength0 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner0.value)
                        
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method.value == "hybrid")
                            hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["f0_file"], open=False):
                            upload_f0_file = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                            f0_file_dropdown = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                            refesh_f0_file = gr.Button(translations["refesh"])
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders.value == "custom")    
                        with gr.Accordion(translations["use_presets"], open=False):
                            with gr.Row():
                                presets_name = gr.Dropdown(label=translations["file_preset"], choices=presets_file, value=presets_file[0] if len(presets_file) > 0 else '', interactive=True, allow_custom_value=True)
                            with gr.Row():
                                load_click = gr.Button(translations["load_file"], variant="primary")
                                refesh_click = gr.Button(translations["refesh"])
                            with gr.Accordion(translations["export_file"], open=False):
                                with gr.Row():
                                    with gr.Column():
                                        with gr.Group():
                                            with gr.Row():
                                                cleaner_chbox = gr.Checkbox(label=translations["save_clean"], value=True, interactive=True)
                                                autotune_chbox = gr.Checkbox(label=translations["save_autotune"], value=True, interactive=True)
                                                pitch_chbox = gr.Checkbox(label=translations["save_pitch"], value=True, interactive=True)
                                                index_strength_chbox = gr.Checkbox(label=translations["save_index_2"], value=True, interactive=True)
                                                resample_sr_chbox = gr.Checkbox(label=translations["save_resample"], value=True, interactive=True)
                                                filter_radius_chbox = gr.Checkbox(label=translations["save_filter"], value=True, interactive=True)
                                                volume_envelope_chbox = gr.Checkbox(label=translations["save_envelope"], value=True, interactive=True)
                                                protect_chbox = gr.Checkbox(label=translations["save_protect"], value=True, interactive=True)
                                                split_audio_chbox = gr.Checkbox(label=translations["save_split"], value=True, interactive=True)
                                                formant_shifting_chbox = gr.Checkbox(label=translations["formantshift"], value=True, interactive=True)
                                with gr.Row():
                                    with gr.Column():
                                        name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
                                        save_file_button = gr.Button(translations["export_file"])
                            with gr.Row():
                                upload_presets = gr.File(label=translations["upload_presets"], file_types=[".json"])  
                        with gr.Column():
                            with gr.Row():
                                split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                                formant_shifting = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                            f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                            resample_sr = gr.Slider(minimum=0, maximum=96000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                convert_button = gr.Button(translations["convert_audio"], variant="primary")
            gr.Markdown(translations["output_convert"])
            with gr.Row():
                main_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["main_convert"])
                backing_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_backing"], visible=convert_backing.value)
                main_backing = gr.Audio(show_download_button=True, interactive=False, label=translations["main_or_backing"], visible=convert_backing.value)  
            with gr.Row():
                original_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_original"], visible=use_original.value)
                vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label=translations["voice_or_instruments"], visible=merge_instrument.value)  
            with gr.Row():
                upload_f0_file.upload(fn=lambda inp: shutil.move(inp.name, os.path.join("assets", "f0")), inputs=[upload_f0_file], outputs=[f0_file_dropdown])
                refesh_f0_file.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown])
                unlock_full_method.change(fn=unlock_f0, inputs=[unlock_full_method], outputs=[method])
            with gr.Row():
                load_click.click(
                    fn=load_presets, 
                    inputs=[
                        presets_name, 
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        formant_qfrency, 
                        formant_timbre
                    ], 
                    outputs=[
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre
                    ]
                )
                refesh_click.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
                save_file_button.click(
                    fn=save_presets, 
                    inputs=[
                        name_to_save_file, 
                        cleaner0, 
                        autotune, 
                        pitch, 
                        clean_strength0, 
                        index_strength, 
                        resample_sr, 
                        filter_radius, 
                        volume_envelope, 
                        protect, 
                        split_audio, 
                        f0_autotune_strength, 
                        cleaner_chbox, 
                        autotune_chbox, 
                        pitch_chbox, 
                        index_strength_chbox, 
                        resample_sr_chbox, 
                        filter_radius_chbox, 
                        volume_envelope_chbox, 
                        protect_chbox, 
                        split_audio_chbox, 
                        formant_shifting_chbox, 
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre
                    ], 
                    outputs=[presets_name]
                )
            with gr.Row():
                upload_presets.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("assets", "presets")), inputs=[upload_presets], outputs=[presets_name])
                autotune.change(fn=visible, inputs=[autotune], outputs=[f0_autotune_strength])
                use_audio.change(fn=lambda a: [visible(a), visible(a), visible(a), visible(a), visible(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), valueFalse_interactive(a), visible(not a), visible(not a), visible(not a), visible(not a)], inputs=[use_audio], outputs=[main_backing, use_original, convert_backing, not_merge_backing, merge_instrument, use_original, convert_backing, not_merge_backing, merge_instrument, input_audio0, output_audio, input0, play_audio])
            with gr.Row():
                convert_backing.change(fn=lambda a,b: [change_backing_choices(a, b), visible(a)], inputs=[convert_backing, not_merge_backing], outputs=[use_original, backing_convert])
                use_original.change(fn=lambda audio, original: [visible(original), visible(not original), visible(audio and not original), valueFalse_interactive(not original), valueFalse_interactive(not original)], inputs=[use_audio, use_original], outputs=[original_convert, main_convert, main_backing, convert_backing, not_merge_backing])
                cleaner0.change(fn=visible, inputs=[cleaner0], outputs=[clean_strength0])
            with gr.Row():
                merge_instrument.change(fn=visible, inputs=[merge_instrument], outputs=[vocal_instrument])
                not_merge_backing.change(fn=lambda audio, merge, cvb: [visible(audio and not merge), change_backing_choices(cvb, merge)], inputs=[use_audio, not_merge_backing, convert_backing], outputs=[main_backing, use_original])
                method.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method, hybrid_method], outputs=[hybrid_method, hop_length])
            with gr.Row():
                hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
                refesh.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
                model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
            with gr.Row():
                input0.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input0], outputs=[input_audio0])
                input_audio0.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio0], outputs=[play_audio])
                formant_shifting.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting], outputs=[formant_qfrency, formant_timbre])
            with gr.Row():
                embedders.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders], outputs=[custom_embedders])
                refesh0.click(fn=change_audios_choices, inputs=[input_audio0], outputs=[input_audio0])
                model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])
            with gr.Row():
                audio_select.change(fn=lambda: visible(True), inputs=[], outputs=[convert_button_2])
                convert_button.click(fn=lambda: visible(False), inputs=[], outputs=[convert_button])
                convert_button_2.click(fn=lambda: [visible(False), visible(False)], inputs=[], outputs=[audio_select, convert_button_2])
            with gr.Row():
                convert_button.click(
                    fn=convert_selection,
                    inputs=[
                        cleaner0,
                        autotune,
                        use_audio,
                        use_original,
                        convert_backing,
                        not_merge_backing,
                        merge_instrument,
                        pitch,
                        clean_strength0,
                        model_pth,
                        model_index,
                        index_strength,
                        input_audio0,
                        output_audio,
                        export_format,
                        method,
                        hybrid_method,
                        hop_length,
                        embedders,
                        custom_embedders,
                        resample_sr,
                        filter_radius,
                        volume_envelope,
                        protect,
                        split_audio,
                        f0_autotune_strength,
                        checkpointing,
                        onnx_f0_mode,
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre,
                        f0_file_dropdown,
                        embed_mode
                    ],
                    outputs=[audio_select, main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
                    api_name="convert_selection"
                )
                embed_mode.change(fn=visible_embedders, inputs=[embed_mode], outputs=[embedders])
                convert_button_2.click(
                    fn=convert_audio,
                    inputs=[
                        cleaner0,
                        autotune,
                        use_audio,
                        use_original,
                        convert_backing,
                        not_merge_backing,
                        merge_instrument,
                        pitch,
                        clean_strength0,
                        model_pth,
                        model_index,
                        index_strength,
                        input_audio0,
                        output_audio,
                        export_format,
                        method,
                        hybrid_method,
                        hop_length,
                        embedders,
                        custom_embedders,
                        resample_sr,
                        filter_radius,
                        volume_envelope,
                        protect,
                        split_audio,
                        f0_autotune_strength,
                        audio_select,
                        checkpointing,
                        onnx_f0_mode,
                        formant_shifting, 
                        formant_qfrency, 
                        formant_timbre,
                        f0_file_dropdown,
                        embed_mode
                    ],
                    outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
                    api_name="convert_audio"
                )


        with gr.TabItem(translations["convert_text"], visible=configs.get("tts_tab", True)):
            gr.Markdown(translations["convert_text_markdown"])
            with gr.Row():
                gr.Markdown(translations["convert_text_markdown_2"])
            with gr.Accordion(translations["model_accordion"], open=True):
                with gr.Row():
                    model_pth0 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                    model_index0 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            use_txt = gr.Checkbox(label=translations["input_txt"], value=False, interactive=True)
                            google_tts_check_box = gr.Checkbox(label=translations["googletts"], value=False, interactive=True)
                        prompt = gr.Textbox(label=translations["text_to_speech"], value="", placeholder="Hello Words", lines=3)
                with gr.Column():
                    speed = gr.Slider(label=translations["voice_speed"], info=translations["voice_speed_info"], minimum=-100, maximum=100, value=0, step=1)
                    pitch0 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
            with gr.Row():
                tts_button = gr.Button(translations["tts_1"], variant="primary", scale=2)
                convert_button0 = gr.Button(translations["tts_2"], variant="secondary", scale=2)
            with gr.Row():
                with gr.Column():
                    txt_input = gr.File(label=translations["drop_text"], file_types=[".txt", ".srt"], visible=use_txt.value)  
                    tts_voice = gr.Dropdown(label=translations["voice"], choices=edgetts, interactive=True, value="vi-VN-NamMinhNeural")
                    tts_pitch = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info_2"], label=translations["pitch"], value=0, interactive=True)
                with gr.Column():
                    refesh1 = gr.Button(translations["refesh"])
                    with gr.Row():
                        index_strength0 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index0.value != "")
                    with gr.Accordion(translations["output_path"], open=False):
                        export_format0 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                        output_audio0 = gr.Textbox(label=translations["output_tts"], value="audios/tts.wav", placeholder="audios/tts.wav", info=translations["tts_output"], interactive=True)
                        output_audio1 = gr.Textbox(label=translations["output_tts_convert"], value="audios/tts-convert.wav", placeholder="audios/tts-convert.wav", info=translations["tts_output"], interactive=True)
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode1 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method3 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method0 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method0 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method0.value == "hybrid")
                            hop_length0 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["f0_file"], open=False):
                            upload_f0_file0 = gr.File(label=translations["upload_f0"], file_types=[".txt"])  
                            f0_file_dropdown0 = gr.Dropdown(label=translations["f0_file_2"], value="", choices=f0_file, allow_custom_value=True, interactive=True)
                            refesh_f0_file0 = gr.Button(translations["refesh"])
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode1 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders0 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders0 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders0.value == "custom")
                        with gr.Group():
                            with gr.Row():
                                formant_shifting1 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)  
                                split_audio0 = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)   
                                cleaner1 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)     
                                autotune3 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True) 
                                checkpointing0 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)         
                        with gr.Column():
                            f0_autotune_strength0 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune3.value)
                            clean_strength1 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner1.value)
                            resample_sr0 = gr.Slider(minimum=0, maximum=96000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope0 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect0 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency1 = gr.Slider(value=1.0, label=translations["formant_qfrency"], info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre1 = gr.Slider(value=1.0, label=translations["formant_timbre"], info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                gr.Markdown(translations["output_tts_markdown"])
            with gr.Row():
                tts_voice_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["output_text_to_speech"])
                tts_voice_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                unlock_full_method3.change(fn=unlock_f0, inputs=[unlock_full_method3], outputs=[method0])
                upload_f0_file0.upload(fn=lambda inp: shutil.move(inp.name, os.path.join("assets", "f0")), inputs=[upload_f0_file0], outputs=[f0_file_dropdown0])
                refesh_f0_file0.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown0])
            with gr.Row():
                embed_mode1.change(fn=visible_embedders, inputs=[embed_mode1], outputs=[embedders0])
                autotune3.change(fn=visible, inputs=[autotune3], outputs=[f0_autotune_strength0])
                model_pth0.change(fn=get_index, inputs=[model_pth0], outputs=[model_index0])
            with gr.Row():
                cleaner1.change(fn=visible, inputs=[cleaner1], outputs=[clean_strength1])
                method0.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method0, hybrid_method0], outputs=[hybrid_method0, hop_length0])
                hybrid_method0.change(fn=hoplength_show, inputs=[method0, hybrid_method0], outputs=[hop_length0])
            with gr.Row():
                refesh1.click(fn=change_models_choices, inputs=[], outputs=[model_pth0, model_index0])
                embedders0.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders0], outputs=[custom_embedders0])
                formant_shifting1.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting1], outputs=[formant_qfrency1, formant_timbre1])
            with gr.Row():
                model_index0.change(fn=index_strength_show, inputs=[model_index0], outputs=[index_strength0])
                txt_input.upload(fn=process_input, inputs=[txt_input], outputs=[prompt])
                use_txt.change(fn=visible, inputs=[use_txt], outputs=[txt_input])
            with gr.Row():
                google_tts_check_box.change(fn=change_tts_voice_choices, inputs=[google_tts_check_box], outputs=[tts_voice])
                tts_button.click(
                    fn=TTS, 
                    inputs=[
                        prompt, 
                        tts_voice, 
                        speed, 
                        output_audio0,
                        tts_pitch,
                        google_tts_check_box,
                        txt_input
                    ], 
                    outputs=[tts_voice_audio],
                    api_name="text-to-speech"
                )
                convert_button0.click(
                    fn=convert_tts,
                    inputs=[
                        cleaner1, 
                        autotune3, 
                        pitch0, 
                        clean_strength1, 
                        model_pth0, 
                        model_index0, 
                        index_strength0, 
                        output_audio0, 
                        output_audio1,
                        export_format0,
                        method0, 
                        hybrid_method0, 
                        hop_length0, 
                        embedders0, 
                        custom_embedders0, 
                        resample_sr0, 
                        filter_radius0, 
                        volume_envelope0, 
                        protect0,
                        split_audio0,
                        f0_autotune_strength0,
                        checkpointing0,
                        onnx_f0_mode1,
                        formant_shifting1, 
                        formant_qfrency1, 
                        formant_timbre1,
                        f0_file_dropdown0,
                        embed_mode1
                    ],
                    outputs=[tts_voice_convert],
                    api_name="convert_tts"
                )

        
        
        with gr.TabItem(translations["downloads"], visible=configs.get("downloads_tab", True)):
            gr.Markdown(translations["download_markdown"])
            with gr.Row():
                gr.Markdown(translations["download_markdown_2"])
            with gr.Row():
                with gr.Accordion(translations["model_download"], open=True):
                    with gr.Row():
                        downloadmodel = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["download_from_csv"], translations["search_models"], translations["upload"]], interactive=True, value=translations["download_url"])
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Column():
                        with gr.Row():
                            url_input = gr.Textbox(label=translations["model_url"], value="", placeholder="https://...", scale=6)
                            download_model_name = gr.Textbox(label=translations["modelname"], value="", placeholder=translations["modelname"], scale=2)
                        url_download = gr.Button(value=translations["downloads"], scale=2)
                    with gr.Column():
                        model_browser = gr.Dropdown(choices=models.keys(), label=translations["model_warehouse"], scale=8, allow_custom_value=True, visible=False)
                        download_from_browser = gr.Button(value=translations["get_model"], scale=2, variant="primary", visible=False)
                    with gr.Column():
                        search_name = gr.Textbox(label=translations["name_to_search"], placeholder=translations["modelname"], interactive=True, scale=8, visible=False)
                        search = gr.Button(translations["search_2"], scale=2, visible=False)
                        search_dropdown = gr.Dropdown(label=translations["select_download_model"], value="", choices=[], allow_custom_value=True, interactive=False, visible=False)
                        download = gr.Button(translations["downloads"], variant="primary", visible=False)
                    with gr.Column():
                        model_upload = gr.File(label=translations["drop_model"], file_types=[".pth", ".onnx", ".index", ".zip"], visible=False)
            with gr.Row():
                with gr.Accordion(translations["download_pretrained_2"], open=False):
                    with gr.Row():
                        pretrain_download_choices = gr.Radio(label=translations["model_download_select"], choices=[translations["download_url"], translations["list_model"], translations["upload"]], value=translations["download_url"], interactive=True)  
                    with gr.Row():
                        gr.Markdown("___")
                    with gr.Column():
                        with gr.Row():
                            pretrainD = gr.Textbox(label=translations["pretrained_url"].format(dg="D"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4)
                            pretrainG = gr.Textbox(label=translations["pretrained_url"].format(dg="G"), value="", info=translations["only_huggingface"], placeholder="https://...", interactive=True, scale=4)
                        download_pretrain_button = gr.Button(translations["downloads"], scale=2)
                    with gr.Column():
                        with gr.Row():
                            pretrain_choices = gr.Dropdown(label=translations["select_pretrain"], info=translations["select_pretrain_info"], choices=list(fetch_pretrained_data().keys()), value="Titan_Medium", allow_custom_value=True, interactive=True, scale=6, visible=False)
                            sample_rate_pretrain = gr.Dropdown(label=translations["pretrain_sr"], info=translations["pretrain_sr"], choices=["48k", "40k", "32k"], value="48k", interactive=True, visible=False)
                        download_pretrain_choices_button = gr.Button(translations["downloads"], scale=2, variant="primary", visible=False)
                    with gr.Row():
                        pretrain_upload_g = gr.File(label=translations["drop_pretrain"].format(dg="G"), file_types=[".pth"], visible=False)
                        pretrain_upload_d = gr.File(label=translations["drop_pretrain"].format(dg="D"), file_types=[".pth"], visible=False)
            with gr.Row():
                url_download.click(
                    fn=download_model, 
                    inputs=[
                        url_input, 
                        download_model_name
                    ], 
                    outputs=[url_input],
                    api_name="download_model"
                )
                download_from_browser.click(
                    fn=lambda model: download_model(models[model], model), 
                    inputs=[model_browser], 
                    outputs=[model_browser],
                    api_name="download_browser"
                )
            with gr.Row():
                downloadmodel.change(fn=change_download_choices, inputs=[downloadmodel], outputs=[url_input, download_model_name, url_download, model_browser, download_from_browser, search_name, search, search_dropdown, download, model_upload])
                search.click(fn=search_models, inputs=[search_name], outputs=[search_dropdown, download])
                model_upload.upload(fn=save_drop_model, inputs=[model_upload], outputs=[model_upload])
                download.click(
                    fn=lambda model: download_model(model_options[model], model), 
                    inputs=[search_dropdown], 
                    outputs=[search_dropdown],
                    api_name="search_models"
                )
            with gr.Row():
                pretrain_download_choices.change(fn=change_download_pretrained_choices, inputs=[pretrain_download_choices], outputs=[pretrainD, pretrainG, download_pretrain_button, pretrain_choices, sample_rate_pretrain, download_pretrain_choices_button, pretrain_upload_d, pretrain_upload_g])
                pretrain_choices.change(fn=update_sample_rate_dropdown, inputs=[pretrain_choices], outputs=[sample_rate_pretrain])
            with gr.Row():
                download_pretrain_button.click(
                    fn=download_pretrained_model,
                    inputs=[
                        pretrain_download_choices, 
                        pretrainD, 
                        pretrainG
                    ],
                    outputs=[pretrainD],
                    api_name="download_pretrain_link"
                )
                download_pretrain_choices_button.click(
                    fn=download_pretrained_model,
                    inputs=[
                        pretrain_download_choices, 
                        pretrain_choices, 
                        sample_rate_pretrain
                    ],
                    outputs=[pretrain_choices],
                    api_name="download_pretrain_choices"
                )
                pretrain_upload_g.upload(
                    fn=lambda pretrain_upload_g: shutil.move(pretrain_upload_g.name, os.path.join("assets", "models", "pretrained_custom")), 
                    inputs=[pretrain_upload_g], 
                    outputs=[],
                    api_name="upload_pretrain_g"
                )
                pretrain_upload_d.upload(
                    fn=lambda pretrain_upload_d: shutil.move(pretrain_upload_d.name, os.path.join("assets", "models", "pretrained_custom")), 
                    inputs=[pretrain_upload_d], 
                    outputs=[],
                    api_name="upload_pretrain_d"
                )

        
        with gr.TabItem(translations["training_model"], visible=configs.get("training_tab", True)):
            gr.Markdown(f"## {translations['training_model']}")
            with gr.Row():
                gr.Markdown(translations["training_markdown"])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            training_name = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                            training_sr = gr.Radio(label=translations["sample_rate"], info=translations["sample_rate_info"], choices=["32k", "40k", "48k"], value="48k", interactive=True) 
                            training_ver = gr.Radio(label=translations["training_version"], info=translations["training_version_info"], choices=["v1", "v2"], value="v2", interactive=True) 
                            with gr.Row():
                                clean_dataset = gr.Checkbox(label=translations["clear_dataset"], value=False, interactive=True)
                                preprocess_cut = gr.Checkbox(label=translations["split_audio"], value=True, interactive=True)
                                process_effects = gr.Checkbox(label=translations["preprocess_effect"], value=False, interactive=True)
                                checkpointing1 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                                training_f0 = gr.Checkbox(label=translations["training_pitch"], value=True, interactive=True)
                                upload = gr.Checkbox(label=translations["upload_dataset"], value=False, interactive=True)
                            with gr.Row():
                                clean_dataset_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.7, step=0.1, interactive=True, visible=clean_dataset.value)
                        with gr.Column():
                            preprocess_button = gr.Button(translations["preprocess_button"], scale=2)
                            upload_dataset = gr.Files(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"], visible=upload.value)
                            preprocess_info = gr.Textbox(label=translations["preprocess_info"], value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(label=translations["f0_method"], open=False):
                                with gr.Group():
                                    with gr.Row():
                                        onnx_f0_mode2 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                        unlock_full_method4 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                    extract_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                                extract_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                            with gr.Accordion(label=translations["hubert_model"], open=False):
                                with gr.Group():
                                    embed_mode2 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                                    extract_embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                                with gr.Row():
                                    extract_embedders_custom = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=extract_embedders.value == "custom")
                        with gr.Column():
                            extract_button = gr.Button(translations["extract_button"], scale=2)
                            extract_info = gr.Textbox(label=translations["extract_info"], value="", interactive=False)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            total_epochs = gr.Slider(label=translations["total_epoch"], info=translations["total_epoch_info"], minimum=1, maximum=10000, value=300, step=1, interactive=True)
                            save_epochs = gr.Slider(label=translations["save_epoch"], info=translations["save_epoch_info"], minimum=1, maximum=10000, value=50, step=1, interactive=True)
                        with gr.Column():
                            with gr.Row():
                                index_button = gr.Button(f"3. {translations['create_index']}", variant="primary", scale=2)
                                training_button = gr.Button(f"4. {translations['training_model']}", variant="primary", scale=2)
                    with gr.Row():
                        with gr.Accordion(label=translations["setting"], open=False):
                            with gr.Row():
                                index_algorithm = gr.Radio(label=translations["index_algorithm"], info=translations["index_algorithm_info"], choices=["Auto", "Faiss", "KMeans"], value="Auto", interactive=True)
                            with gr.Row():
                                custom_dataset = gr.Checkbox(label=translations["custom_dataset"], info=translations["custom_dataset_info"], value=False, interactive=True)
                                overtraining_detector = gr.Checkbox(label=translations["overtraining_detector"], info=translations["overtraining_detector_info"], value=False, interactive=True)
                                clean_up = gr.Checkbox(label=translations["cleanup_training"], info=translations["cleanup_training_info"], value=False, interactive=True)
                                cache_in_gpu = gr.Checkbox(label=translations["cache_in_gpu"], info=translations["cache_in_gpu_info"], value=False, interactive=True)
                            with gr.Column():
                                dataset_path = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True, visible=custom_dataset.value)
                            with gr.Column():
                                threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label=translations["threshold"], interactive=True, visible=overtraining_detector.value)
                                with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                                    with gr.Column():
                                        gpu_number = gr.Textbox(label=translations["gpu_number"], value=str("-".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else "-"), info=translations["gpu_number_info"], interactive=True)
                                        gpu_info = gr.Textbox(label=translations["gpu_info"], value=get_gpu_info(), info=translations["gpu_info_2"], interactive=False)
                                        cpu_core = gr.Slider(label=translations["cpu_core"], info=translations["cpu_core_info"], minimum=0, maximum=cpu_count(), value=cpu_count(), step=1, interactive=True)          
                                        train_batch_size = gr.Slider(label=translations["batch_size"], info=translations["batch_size_info"], minimum=1, maximum=64, value=8, step=1, interactive=True)
                            with gr.Row():
                                save_only_latest = gr.Checkbox(label=translations["save_only_latest"], info=translations["save_only_latest_info"], value=True, interactive=True)
                                save_every_weights = gr.Checkbox(label=translations["save_every_weights"], info=translations["save_every_weights_info"], value=True, interactive=True)
                                not_use_pretrain = gr.Checkbox(label=translations["not_use_pretrain_2"], info=translations["not_use_pretrain_info"], value=False, interactive=True)
                                custom_pretrain = gr.Checkbox(label=translations["custom_pretrain"], info=translations["custom_pretrain_info"], value=False, interactive=True)
                            with gr.Row():
                                vocoders = gr.Radio(label=translations["vocoder"], info=translations["vocoder_info"], choices=["Default", "MRF-HiFi-GAN", "RefineGAN"], value="Default", interactive=True) 
                            with gr.Row():
                                deterministic = gr.Checkbox(label=translations["deterministic"], info=translations["deterministic_info"], value=False, interactive=True)
                                benchmark = gr.Checkbox(label=translations["benchmark"], info=translations["benchmark_info"], value=False, interactive=True)
                            with gr.Row():
                                model_author = gr.Textbox(label=translations["training_author"], info=translations["training_author_info"], value="", placeholder=translations["training_author"], interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    with gr.Accordion(translations["custom_pretrain_info"], open=False, visible=custom_pretrain.value and not not_use_pretrain.value) as pretrain_setting:
                                        pretrained_D = gr.Dropdown(label=translations["pretrain_file"].format(dg="D"), choices=pretrainedD, value=pretrainedD[0] if len(pretrainedD) > 0 else '', interactive=True, allow_custom_value=True)
                                        pretrained_G = gr.Dropdown(label=translations["pretrain_file"].format(dg="G"), choices=pretrainedG, value=pretrainedG[0] if len(pretrainedG) > 0 else '', interactive=True, allow_custom_value=True)
                                        refesh_pretrain = gr.Button(translations["refesh"], scale=2)
                    with gr.Row():
                        training_info = gr.Textbox(label=translations["train_info"], value="", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(translations["export_model"], open=False):
                                with gr.Row():
                                    model_file= gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                                    index_file = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                                with gr.Row():
                                    refesh_file = gr.Button(f"1. {translations['refesh']}", scale=2)
                                    zip_model = gr.Button(translations["zip_model"], variant="primary", scale=2)
                                with gr.Row():
                                    zip_output = gr.File(label=translations["output_zip"], file_types=[".zip"], interactive=False, visible=False)
            with gr.Row():
                vocoders.change(fn=pitch_guidance_lock, inputs=[vocoders], outputs=[training_f0])
                training_f0.change(fn=vocoders_lock, inputs=[training_f0, vocoders], outputs=[vocoders])
                unlock_full_method4.change(fn=unlock_f0, inputs=[unlock_full_method4], outputs=[extract_method])
            with gr.Row():
                refesh_file.click(fn=change_models_choices, inputs=[], outputs=[model_file, index_file]) 
                zip_model.click(fn=zip_file, inputs=[training_name, model_file, index_file], outputs=[zip_output])                
                dataset_path.change(fn=lambda folder: os.makedirs(folder, exist_ok=True), inputs=[dataset_path], outputs=[])
            with gr.Row():
                upload.change(fn=visible, inputs=[upload], outputs=[upload_dataset]) 
                overtraining_detector.change(fn=visible, inputs=[overtraining_detector], outputs=[threshold]) 
                clean_dataset.change(fn=visible, inputs=[clean_dataset], outputs=[clean_dataset_strength])
            with gr.Row():
                custom_dataset.change(fn=lambda custom_dataset: [visible(custom_dataset), "dataset"],inputs=[custom_dataset], outputs=[dataset_path, dataset_path])
                training_ver.change(fn=unlock_vocoder, inputs=[training_ver, vocoders], outputs=[vocoders])
                vocoders.change(fn=unlock_ver, inputs=[training_ver, vocoders], outputs=[training_ver])
                upload_dataset.upload(
                    fn=lambda files, folder: [shutil.move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr_warning(translations["dataset_folder1"]),
                    inputs=[upload_dataset, dataset_path], 
                    outputs=[], 
                    api_name="upload_dataset"
                )           
            with gr.Row():
                not_use_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
                custom_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
                refesh_pretrain.click(fn=change_pretrained_choices, inputs=[], outputs=[pretrained_D, pretrained_G])
            with gr.Row():
                preprocess_button.click(
                    fn=preprocess,
                    inputs=[
                        training_name, 
                        training_sr, 
                        cpu_core,
                        preprocess_cut, 
                        process_effects,
                        dataset_path,
                        clean_dataset,
                        clean_dataset_strength
                    ],
                    outputs=[preprocess_info],
                    api_name="preprocess"
                )
            with gr.Row():
                embed_mode2.change(fn=visible_embedders, inputs=[embed_mode2], outputs=[extract_embedders])
                extract_method.change(fn=hoplength_show, inputs=[extract_method], outputs=[extract_hop_length])
                extract_embedders.change(fn=lambda extract_embedders: visible(extract_embedders == "custom"), inputs=[extract_embedders], outputs=[extract_embedders_custom])
            with gr.Row():
                extract_button.click(
                    fn=extract,
                    inputs=[
                        training_name, 
                        training_ver, 
                        extract_method, 
                        training_f0, 
                        extract_hop_length, 
                        cpu_core,
                        gpu_number,
                        training_sr, 
                        extract_embedders, 
                        extract_embedders_custom,
                        onnx_f0_mode2,
                        embed_mode2
                    ],
                    outputs=[extract_info],
                    api_name="extract"
                )
            with gr.Row():
                index_button.click(
                    fn=create_index,
                    inputs=[
                        training_name, 
                        training_ver, 
                        index_algorithm
                    ],
                    outputs=[training_info],
                    api_name="create_index"
                )
            with gr.Row():
                training_button.click(
                    fn=training,
                    inputs=[
                        training_name, 
                        training_ver, 
                        save_epochs, 
                        save_only_latest, 
                        save_every_weights, 
                        total_epochs, 
                        training_sr,
                        train_batch_size, 
                        gpu_number,
                        training_f0,
                        not_use_pretrain,
                        custom_pretrain,
                        pretrained_G,
                        pretrained_D,
                        overtraining_detector,
                        threshold,
                        clean_up,
                        cache_in_gpu,
                        model_author,
                        vocoders,
                        checkpointing1,
                        deterministic, 
                        benchmark
                    ],
                    outputs=[training_info],
                    api_name="training_model"
                )

        with gr.TabItem(translations["convert_with_whisper"], visible=configs.get("convert_with_whisper", True)):
            gr.Markdown(f"## {translations['convert_with_whisper']}")
            with gr.Row():
                gr.Markdown(translations["convert_with_whisper_info"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            cleaner2 = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            autotune2 = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            checkpointing2 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                            formant_shifting2 = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                        with gr.Row():
                            num_spk = gr.Slider(minimum=2, maximum=8, step=1, info=translations["num_spk_info"], label=translations["num_spk"], value=2, interactive=True)
            with gr.Row():
                with gr.Column():
                    convert_button3 = gr.Button(translations["convert_audio"], variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"] + " 1", open=True):
                        with gr.Row():
                            model_pth2 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index2 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh2 = gr.Button(translations["refesh"])
                        with gr.Row():
                            pitch3 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                            index_strength2 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index2.value != "")
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_format2 = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audio1 = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audio2 = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh4 = gr.Button(translations["refesh"])
                        with gr.Row():
                            input2 = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                with gr.Column():
                    with gr.Accordion(translations["model_accordion"] + " 2", open=True):
                        with gr.Row():
                            model_pth3 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            model_index3 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refesh3 = gr.Button(translations["refesh"])
                        with gr.Row():
                            pitch4 = gr.Slider(minimum=-20, maximum=20, step=1, info=translations["pitch_info"], label=translations["pitch"], value=0, interactive=True)
                            index_strength3 = gr.Slider(label=translations["index_strength"], info=translations["index_strength_info"], minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, visible=model_index3.value != "")
                    with gr.Accordion(translations["setting"], open=False):
                        with gr.Row():
                            model_size = gr.Radio(label=translations["model_size"], info=translations["model_size_info"], choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], value="medium", interactive=True)
                        with gr.Accordion(translations["f0_method"], open=False):
                            with gr.Group():
                                with gr.Row():
                                    onnx_f0_mode4 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                                    unlock_full_method2 = gr.Checkbox(label=translations["f0_unlock"], info=translations["f0_unlock_info"], value=False, interactive=True)
                                method3 = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0+["hybrid"], value="rmvpe", interactive=True)
                                hybrid_method3 = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=["hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]", "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]", "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]", "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]", "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]", "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]", "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]", "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]", "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]", "hybrid[harvest+yin]"], value="hybrid[pm+dio]", interactive=True, allow_custom_value=True, visible=method3.value == "hybrid")
                            hop_length3 = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=512, value=128, step=1, interactive=True, visible=False)
                        with gr.Accordion(translations["hubert_model"], open=False):
                            embed_mode3 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            embedders3 = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                            custom_embedders3 = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=embedders3.value == "custom")
                        with gr.Column():      
                            clean_strength3 = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.5, step=0.1, interactive=True, visible=cleaner2.value)
                            f0_autotune_strength3 = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                            resample_sr3 = gr.Slider(minimum=0, maximum=96000, label=translations["resample"], info=translations["resample_info"], value=0, step=1, interactive=True)
                            filter_radius3 = gr.Slider(minimum=0, maximum=7, label=translations["filter_radius"], info=translations["filter_radius_info"], value=3, step=1, interactive=True)
                            volume_envelope3 = gr.Slider(minimum=0, maximum=1, label=translations["volume_envelope"], info=translations["volume_envelope_info"], value=1, step=0.1, interactive=True)
                            protect3 = gr.Slider(minimum=0, maximum=1, label=translations["protect"], info=translations["protect_info"], value=0.5, step=0.01, interactive=True)
                        with gr.Row():
                            formant_qfrency3 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 1", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre3 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 1", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                        with gr.Row():
                            formant_qfrency4 = gr.Slider(value=1.0, label=translations["formant_qfrency"] + " 2", info=translations["formant_qfrency"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
                            formant_timbre4 = gr.Slider(value=1.0, label=translations["formant_timbre"] + " 2", info=translations["formant_timbre"], minimum=0.0, maximum=16.0, step=0.1, interactive=True, visible=False)
            with gr.Row():
                gr.Markdown(translations["input_output"])
            with gr.Row():
                play_audio2 = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                play_audio3 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_file_tts_convert"])
            with gr.Row():
                autotune2.change(fn=visible, inputs=[autotune2], outputs=[f0_autotune_strength3])
                cleaner2.change(fn=visible, inputs=[cleaner2], outputs=[clean_strength3])
                method3.change(fn=lambda method, hybrid: [visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[method3, hybrid_method3], outputs=[hybrid_method3, hop_length3])
            with gr.Row():
                hybrid_method3.change(fn=hoplength_show, inputs=[method3, hybrid_method3], outputs=[hop_length3])
                refesh2.click(fn=change_models_choices, inputs=[], outputs=[model_pth2, model_index2])
                model_pth2.change(fn=get_index, inputs=[model_pth2], outputs=[model_index2])
            with gr.Row():
                refesh3.click(fn=change_models_choices, inputs=[], outputs=[model_pth3, model_index3])
                model_pth3.change(fn=get_index, inputs=[model_pth3], outputs=[model_index3])
                input2.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[input2], outputs=[input_audio1])
            with gr.Row():
                input_audio1.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio1], outputs=[play_audio2])
                formant_shifting2.change(fn=lambda a: [visible(a)]*4, inputs=[formant_shifting2], outputs=[formant_qfrency3, formant_timbre3, formant_qfrency4, formant_timbre4])
                embedders3.change(fn=lambda embedders: visible(embedders == "custom"), inputs=[embedders3], outputs=[custom_embedders3])
            with gr.Row():
                refesh4.click(fn=change_audios_choices, inputs=[input_audio1], outputs=[input_audio1])
                model_index2.change(fn=index_strength_show, inputs=[model_index2], outputs=[index_strength2])
                model_index3.change(fn=index_strength_show, inputs=[model_index3], outputs=[index_strength3])
            with gr.Row():
                unlock_full_method2.change(fn=unlock_f0, inputs=[unlock_full_method2], outputs=[method3])
                embed_mode3.change(fn=visible_embedders, inputs=[embed_mode3], outputs=[embedders3])
                convert_button3.click(
                    fn=convert_with_whisper,
                    inputs=[
                        num_spk,
                        model_size,
                        cleaner2,
                        clean_strength3,
                        autotune2,
                        f0_autotune_strength3,
                        checkpointing2,
                        model_pth2,
                        model_pth3,
                        model_index2,
                        model_index3,
                        pitch3,
                        pitch4,
                        index_strength2,
                        index_strength3,
                        export_format2,
                        input_audio1,
                        output_audio2,
                        onnx_f0_mode4,
                        method3,
                        hybrid_method3,
                        hop_length3,
                        embed_mode3,
                        embedders3,
                        custom_embedders3,
                        resample_sr3,
                        filter_radius3,
                        volume_envelope3,
                        protect3,
                        formant_shifting2,
                        formant_qfrency3,
                        formant_timbre3,
                        formant_qfrency4,
                        formant_timbre4,
                    ],
                    outputs=[play_audio3],
                    api_name="convert_with_whisper"
                )

        
        with gr.TabItem(translations["audio_editing"], visible=configs.get("audioldm2", True)):
            gr.Markdown(translations["audio_editing_info"])
            with gr.Row():
                gr.Markdown(translations["audio_editing_markdown"])
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            save_compute = gr.Checkbox(label=translations["save_compute"], value=True, interactive=True)
                        tar_prompt = gr.Textbox(label=translations["target_prompt"], info=translations["target_prompt_info"], placeholder="Piano and violin cover", lines=5, interactive=True)
                with gr.Column():
                    cfg_scale_src = gr.Slider(value=3, minimum=0.5, maximum=25, label=translations["cfg_scale_src"], info=translations["cfg_scale_src_info"], interactive=True)
                    cfg_scale_tar = gr.Slider(value=12, minimum=0.5, maximum=25, label=translations["cfg_scale_tar"], info=translations["cfg_scale_tar_info"], interactive=True)
            with gr.Row():
                edit_button = gr.Button(translations["editing"], variant="primary")
            with gr.Row():
                with gr.Column():
                    drop_audio_file = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])  
                    display_audio = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["input_output"], open=False):
                        with gr.Column():
                            export_audio_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
                            input_audiopath = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, info=translations["provide_audio"], allow_custom_value=True, interactive=True)
                            output_audiopath = gr.Textbox(label=translations["output_path"], value="audios/output.wav", placeholder="audios/output.wav", info=translations["output_path_info"], interactive=True)
                        with gr.Column():
                            refesh_audio = gr.Button(translations["refesh"])
                    with gr.Accordion(translations["setting"], open=False):
                        audioldm2_model = gr.Radio(label=translations["audioldm2_model"], info=translations["audioldm2_model_info"], choices=["audioldm2", "audioldm2-large", "audioldm2-music"], value="audioldm2-music", interactive=True)
                        with gr.Row():
                            src_prompt = gr.Textbox(label=translations["source_prompt"], lines=2, interactive=True, info=translations["source_prompt_info"], placeholder="A recording of a happy upbeat classical music piece")
                        with gr.Row():
                            with gr.Column(): 
                                audioldm2_sample_rate = gr.Slider(minimum=8000, maximum=96000, label=translations["sr"], info=translations["sr_info"], value=44100, step=1, interactive=True)
                                t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label=translations["t_start"], interactive=True, info=translations["t_start_info"])
                                steps = gr.Slider(value=50, step=1, minimum=10, maximum=300, label=translations["steps_label"], info=translations["steps_info"], interactive=True)
            with gr.Row():
                gr.Markdown(translations["output_audio"])
            with gr.Row():
                output_audioldm2 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
            with gr.Row():
                refesh_audio.click(fn=change_audios_choices, inputs=[input_audiopath], outputs=[input_audiopath])
                drop_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[drop_audio_file], outputs=[input_audiopath])
                input_audiopath.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audiopath], outputs=[display_audio])
            with gr.Row():
                edit_button.click(
                    fn=run_audioldm2,
                    inputs=[
                        input_audiopath, 
                        output_audiopath, 
                        export_audio_format, 
                        audioldm2_sample_rate, 
                        audioldm2_model, 
                        src_prompt, 
                        tar_prompt, 
                        steps, 
                        cfg_scale_src, 
                        cfg_scale_tar, 
                        t_start, 
                        save_compute
                    ],
                    outputs=[output_audioldm2],
                    api_name="audioldm2"
                )

        with gr.TabItem(translations["audio_effects"], visible=configs.get("effects_tab", True)):
            gr.Markdown(translations["apply_audio_effects"])
            with gr.Row():
                gr.Markdown(translations["audio_effects_edit"])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        reverb_check_box = gr.Checkbox(label=translations["reverb"], value=False, interactive=True)
                        chorus_check_box = gr.Checkbox(label=translations["chorus"], value=False, interactive=True)
                        delay_check_box = gr.Checkbox(label=translations["delay"], value=False, interactive=True)
                        phaser_check_box = gr.Checkbox(label=translations["phaser"], value=False, interactive=True)
                        compressor_check_box = gr.Checkbox(label=translations["compressor"], value=False, interactive=True)
                        more_options = gr.Checkbox(label=translations["more_option"], value=False, interactive=True)    
            with gr.Row():
                with gr.Accordion(translations["input_output"], open=False):
                    with gr.Row():
                        upload_audio = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                    with gr.Row():
                        audio_in_path = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True)
                        audio_out_path = gr.Textbox(label=translations["output_audio"], value="audios/audio_effects.wav", placeholder="audios/audio_effects.wav", info=translations["provide_output"], interactive=True)
                    with gr.Row():
                        with gr.Column():
                            audio_combination = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True)
                            audio_combination_input = gr.Dropdown(label=translations["input_audio"], value="", choices=paths_for_files, info=translations["provide_audio"], interactive=True, allow_custom_value=True, visible=audio_combination.value)
                    with gr.Row():
                        audio_effects_refesh = gr.Button(translations["refesh"])
                    with gr.Row():
                        audio_output_format = gr.Radio(label=translations["export_format"], info=translations["export_info"], choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"], value="wav", interactive=True)
            with gr.Row():
                apply_effects_button = gr.Button(translations["apply"], variant="primary", scale=2)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["reverb"], open=False, visible=reverb_check_box.value) as reverb_accordion:
                            reverb_freeze_mode = gr.Checkbox(label=translations["reverb_freeze"], info=translations["reverb_freeze_info"], value=False, interactive=True)
                            reverb_room_size = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.15, label=translations["room_size"], info=translations["room_size_info"], interactive=True)
                            reverb_damping = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label=translations["damping"], info=translations["damping_info"], interactive=True)
                            reverb_wet_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label=translations["wet_level"], info=translations["wet_level_info"], interactive=True)
                            reverb_dry_level = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.8, label=translations["dry_level"], info=translations["dry_level_info"], interactive=True)
                            reverb_width = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label=translations["width"], info=translations["width_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["chorus"], open=False, visible=chorus_check_box.value) as chorus_accordion:
                            chorus_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_depth"], info=translations["chorus_depth_info"], interactive=True)
                            chorus_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.5, label=translations["chorus_rate_hz"], info=translations["chorus_rate_hz_info"], interactive=True)
                            chorus_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["chorus_mix"], info=translations["chorus_mix_info"], interactive=True)
                            chorus_centre_delay_ms = gr.Slider(minimum=0, maximum=50, step=1, value=10, label=translations["chorus_centre_delay_ms"], info=translations["chorus_centre_delay_ms_info"], interactive=True)
                            chorus_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["chorus_feedback"], info=translations["chorus_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["delay"], open=False, visible=delay_check_box.value) as delay_accordion:
                            delay_second = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.5, label=translations["delay_seconds"], info=translations["delay_seconds_info"], interactive=True)
                            delay_feedback = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_feedback"], info=translations["delay_feedback_info"], interactive=True)
                            delay_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["delay_mix"], info=translations["delay_mix_info"], interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(translations["more_option"], open=False, visible=more_options.value) as more_accordion:
                            with gr.Row():
                                fade = gr.Checkbox(label=translations["fade"], value=False, interactive=True)
                                bass_or_treble = gr.Checkbox(label=translations["bass_or_treble"], value=False, interactive=True)
                                limiter = gr.Checkbox(label=translations["limiter"], value=False, interactive=True)
                                resample_checkbox = gr.Checkbox(label=translations["resample"], value=False, interactive=True)
                            with gr.Row():
                                distortion_checkbox = gr.Checkbox(label=translations["distortion"], value=False, interactive=True)
                                gain_checkbox = gr.Checkbox(label=translations["gain"], value=False, interactive=True)
                                bitcrush_checkbox = gr.Checkbox(label=translations["bitcrush"], value=False, interactive=True)
                                clipping_checkbox = gr.Checkbox(label=translations["clipping"], value=False, interactive=True)
                            with gr.Accordion(translations["fade"], open=True, visible=fade.value) as fade_accordion:
                                with gr.Row():
                                    fade_in = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_in"], info=translations["fade_in_info"], interactive=True)
                                    fade_out = gr.Slider(minimum=0, maximum=10000, step=100, value=0, label=translations["fade_out"], info=translations["fade_out_info"], interactive=True)
                            with gr.Accordion(translations["bass_or_treble"], open=True, visible=bass_or_treble.value) as bass_treble_accordion:
                                with gr.Row():
                                    bass_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["bass_boost"], info=translations["bass_boost_info"], interactive=True)
                                    bass_frequency = gr.Slider(minimum=20, maximum=200, step=10, value=100, label=translations["bass_frequency"], info=translations["bass_frequency_info"], interactive=True)
                                with gr.Row():
                                    treble_boost = gr.Slider(minimum=0, maximum=20, step=1, value=0, label=translations["treble_boost"], info=translations["treble_boost_info"], interactive=True)
                                    treble_frequency = gr.Slider(minimum=1000, maximum=10000, step=500, value=3000, label=translations["treble_frequency"], info=translations["treble_frequency_info"], interactive=True)
                            with gr.Accordion(translations["limiter"], open=True, visible=limiter.value) as limiter_accordion:
                                with gr.Row():
                                    limiter_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["limiter_threashold_db"], info=translations["limiter_threashold_db_info"], interactive=True)
                                    limiter_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["limiter_release_ms"], info=translations["limiter_release_ms_info"], interactive=True)
                            with gr.Column():
                                pitch_shift_semitones = gr.Slider(minimum=-20, maximum=20, step=1, value=0, label=translations["pitch"], info=translations["pitch_info"], interactive=True)
                                audio_effect_resample_sr = gr.Slider(minimum=0, maximum=96000, step=1, value=0, label=translations["resample"], info=translations["resample_info"], interactive=True, visible=resample_checkbox.value)
                                distortion_drive_db = gr.Slider(minimum=0, maximum=50, step=1, value=20, label=translations["distortion"], info=translations["distortion_info"], interactive=True, visible=distortion_checkbox.value)
                                gain_db = gr.Slider(minimum=-60, maximum=60, step=1, value=0, label=translations["gain"], info=translations["gain_info"], interactive=True, visible=gain_checkbox.value)
                                clipping_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-1, label=translations["clipping_threashold_db"], info=translations["clipping_threashold_db_info"], interactive=True, visible=clipping_checkbox.value)
                                bitcrush_bit_depth = gr.Slider(minimum=1, maximum=24, step=1, value=16, label=translations["bitcrush_bit_depth"], info=translations["bitcrush_bit_depth_info"], interactive=True, visible=bitcrush_checkbox.value)
                    with gr.Row():
                        with gr.Accordion(translations["phaser"], open=False, visible=phaser_check_box.value) as phaser_accordion:
                            phaser_depth = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_depth"], info=translations["phaser_depth_info"], interactive=True)
                            phaser_rate_hz = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label=translations["phaser_rate_hz"], info=translations["phaser_rate_hz_info"], interactive=True)
                            phaser_mix = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label=translations["phaser_mix"], info=translations["phaser_mix_info"], interactive=True)
                            phaser_centre_frequency_hz = gr.Slider(minimum=50, maximum=5000, step=10, value=1000, label=translations["phaser_centre_frequency_hz"], info=translations["phaser_centre_frequency_hz_info"], interactive=True)
                            phaser_feedback = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label=translations["phaser_feedback"], info=translations["phaser_feedback_info"], interactive=True)
                    with gr.Row():
                        with gr.Accordion(translations["compressor"], open=False, visible=compressor_check_box.value) as compressor_accordion:
                            compressor_threashold_db = gr.Slider(minimum=-60, maximum=0, step=1, value=-20, label=translations["compressor_threashold_db"], info=translations["compressor_threashold_db_info"], interactive=True)
                            compressor_ratio = gr.Slider(minimum=1, maximum=20, step=0.1, value=1, label=translations["compressor_ratio"], info=translations["compressor_ratio_info"], interactive=True)
                            compressor_attack_ms = gr.Slider(minimum=0.1, maximum=100, step=0.1, value=10, label=translations["compressor_attack_ms"], info=translations["compressor_attack_ms_info"], interactive=True)
                            compressor_release_ms = gr.Slider(minimum=10, maximum=1000, step=1, value=100, label=translations["compressor_release_ms"], info=translations["compressor_release_ms_info"], interactive=True)   
            with gr.Row():
                gr.Markdown(translations["output_audio"])
            with gr.Row():
                audio_play_input = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                audio_play_output = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
            with gr.Row():
                reverb_check_box.change(fn=visible, inputs=[reverb_check_box], outputs=[reverb_accordion])
                chorus_check_box.change(fn=visible, inputs=[chorus_check_box], outputs=[chorus_accordion])
                delay_check_box.change(fn=visible, inputs=[delay_check_box], outputs=[delay_accordion])
            with gr.Row():
                compressor_check_box.change(fn=visible, inputs=[compressor_check_box], outputs=[compressor_accordion])
                phaser_check_box.change(fn=visible, inputs=[phaser_check_box], outputs=[phaser_accordion])
                more_options.change(fn=visible, inputs=[more_options], outputs=[more_accordion])
            with gr.Row():
                fade.change(fn=visible, inputs=[fade], outputs=[fade_accordion])
                bass_or_treble.change(fn=visible, inputs=[bass_or_treble], outputs=[bass_treble_accordion])
                limiter.change(fn=visible, inputs=[limiter], outputs=[limiter_accordion])
                resample_checkbox.change(fn=visible, inputs=[resample_checkbox], outputs=[audio_effect_resample_sr])
            with gr.Row():
                distortion_checkbox.change(fn=visible, inputs=[distortion_checkbox], outputs=[distortion_drive_db])
                gain_checkbox.change(fn=visible, inputs=[gain_checkbox], outputs=[gain_db])
                clipping_checkbox.change(fn=visible, inputs=[clipping_checkbox], outputs=[clipping_threashold_db])
                bitcrush_checkbox.change(fn=visible, inputs=[bitcrush_checkbox], outputs=[bitcrush_bit_depth])
            with gr.Row():
                upload_audio.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio], outputs=[audio_in_path])
                audio_in_path.change(fn=lambda audio: audio if audio else None, inputs=[audio_in_path], outputs=[audio_play_input])
                audio_effects_refesh.click(fn=lambda a, b: [change_audios_choices(a), change_audios_choices(b)], inputs=[audio_in_path, audio_combination_input], outputs=[audio_in_path, audio_combination_input])
            with gr.Row():
                more_options.change(fn=lambda: [False]*8, inputs=[], outputs=[fade, bass_or_treble, limiter, resample_checkbox, distortion_checkbox, gain_checkbox, clipping_checkbox, bitcrush_checkbox])
                audio_combination.change(fn=visible, inputs=[audio_combination], outputs=[audio_combination_input])
            with gr.Row():
                apply_effects_button.click(
                    fn=audio_effects,
                    inputs=[
                        audio_in_path, 
                        audio_out_path, 
                        resample_checkbox, 
                        audio_effect_resample_sr, 
                        chorus_depth, 
                        chorus_rate_hz, 
                        chorus_mix, 
                        chorus_centre_delay_ms, 
                        chorus_feedback, 
                        distortion_drive_db, 
                        reverb_room_size, 
                        reverb_damping, 
                        reverb_wet_level, 
                        reverb_dry_level, 
                        reverb_width, 
                        reverb_freeze_mode, 
                        pitch_shift_semitones, 
                        delay_second, 
                        delay_feedback, 
                        delay_mix, 
                        compressor_threashold_db, 
                        compressor_ratio, 
                        compressor_attack_ms, 
                        compressor_release_ms, 
                        limiter_threashold_db, 
                        limiter_release_ms, 
                        gain_db, 
                        bitcrush_bit_depth, 
                        clipping_threashold_db, 
                        phaser_rate_hz, 
                        phaser_depth, 
                        phaser_centre_frequency_hz, 
                        phaser_feedback, 
                        phaser_mix, 
                        bass_boost, 
                        bass_frequency, 
                        treble_boost, 
                        treble_frequency, 
                        fade_in, 
                        fade_out, 
                        audio_output_format, 
                        chorus_check_box, 
                        distortion_checkbox, 
                        reverb_check_box, 
                        delay_check_box, 
                        compressor_check_box, 
                        limiter, 
                        gain_checkbox, 
                        bitcrush_checkbox, 
                        clipping_checkbox, 
                        phaser_check_box, 
                        bass_or_treble, 
                        fade,
                        audio_combination,
                        audio_combination_input
                    ],
                    outputs=[audio_play_output],
                    api_name="audio_effects"
                )

        with gr.TabItem(translations["createdataset"], visible=configs.get("create_dataset_tab", True)):
            gr.Markdown(translations["create_dataset_markdown"])
            with gr.Row():
                gr.Markdown(translations["create_dataset_markdown_2"])
            with gr.Row():
                dataset_url = gr.Textbox(label=translations["url_audio"], info=translations["create_dataset_url"], value="", placeholder="https://www.youtube.com/...", interactive=True)
                output_dataset = gr.Textbox(label=translations["output_data"], info=translations["output_data_info"], value="dataset", placeholder="dataset", interactive=True)
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            separator_reverb = gr.Checkbox(label=translations["dereveb_audio"], value=False, interactive=True)
                            denoise_mdx = gr.Checkbox(label=translations["denoise"], value=False, interactive=True)
                        with gr.Row():
                            kim_vocal_version = gr.Radio(label=translations["model_ver"], info=translations["model_ver_info"], choices=["Version-1", "Version-2"], value="Version-2", interactive=True)
                            kim_vocal_overlap = gr.Radio(label=translations["overlap"], info=translations["overlap_info"], choices=["0.25", "0.5", "0.75", "0.99"], value="0.25", interactive=True)
                        with gr.Row():    
                            kim_vocal_hop_length = gr.Slider(label="Hop length", info=translations["hop_length_info"], minimum=1, maximum=8192, value=1024, step=1, interactive=True)
                            kim_vocal_batch_size = gr.Slider(label=translations["batch_size"], info=translations["mdx_batch_size_info"], minimum=1, maximum=64, value=1, step=1, interactive=True) 
                        with gr.Row():
                            kim_vocal_segments_size = gr.Slider(label=translations["segments_size"], info=translations["segments_size_info"], minimum=32, maximum=3072, value=256, step=32, interactive=True)
                        with gr.Row():
                            sample_rate0 = gr.Slider(minimum=8000, maximum=96000, step=1, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
                with gr.Column():
                    create_button = gr.Button(translations["createdataset"], variant="primary", scale=2, min_width=4000)
                    with gr.Group():
                        with gr.Row():
                            clean_audio = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                            skip = gr.Checkbox(label=translations["skip"], value=False, interactive=True)
                        with gr.Row():   
                            dataset_clean_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label=translations["clean_strength"], info=translations["clean_strength_info"], interactive=True, visible=clean_audio.value)
                        with gr.Row():
                            skip_start = gr.Textbox(label=translations["skip_start"], info=translations["skip_start_info"], value="", placeholder="0,...", interactive=True, visible=skip.value)
                            skip_end = gr.Textbox(label=translations["skip_end"], info=translations["skip_end_info"], value="", placeholder="0,...", interactive=True, visible=skip.value)
                    create_dataset_info = gr.Textbox(label=translations["create_dataset_info"], value="", interactive=False)
            with gr.Row():
                clean_audio.change(fn=visible, inputs=[clean_audio], outputs=[dataset_clean_strength])
                skip.change(fn=lambda a: [valueEmpty_visible1(a)]*2, inputs=[skip], outputs=[skip_start, skip_end])
            with gr.Row():
                create_button.click(
                    fn=create_dataset,
                    inputs=[
                        dataset_url, 
                        output_dataset, 
                        clean_audio, 
                        dataset_clean_strength, 
                        separator_reverb, 
                        kim_vocal_version, 
                        kim_vocal_overlap, 
                        kim_vocal_segments_size, 
                        denoise_mdx, 
                        skip, 
                        skip_start, 
                        skip_end,
                        kim_vocal_hop_length,
                        kim_vocal_batch_size,
                        sample_rate0
                    ],
                    outputs=[create_dataset_info],
                    api_name="create_dataset"
                )

        with gr.TabItem(translations["fushion"], visible=configs.get("fushion_tab", True)):
            gr.Markdown(translations["fushion_markdown"])
            with gr.Row():
                gr.Markdown(translations["fushion_markdown_2"])
            with gr.Row():
                name_to_save = gr.Textbox(label=translations["modelname"], placeholder="Model.pth", value="", max_lines=1, interactive=True)
            with gr.Row():
                fushion_button = gr.Button(translations["fushion"], variant="primary", scale=4)
            with gr.Column():
                with gr.Row():
                    model_a = gr.File(label=f"{translations['model_name']} 1", file_types=[".pth", ".onnx"]) 
                    model_b = gr.File(label=f"{translations['model_name']} 2", file_types=[".pth", ".onnx"])
                with gr.Row():
                    model_path_a = gr.Textbox(label=f"{translations['model_path']} 1", value="", placeholder="assets/weights/Model_1.pth")
                    model_path_b = gr.Textbox(label=f"{translations['model_path']} 2", value="", placeholder="assets/weights/Model_2.pth")
            with gr.Row():
                ratio = gr.Slider(minimum=0, maximum=1, label=translations["model_ratio"], info=translations["model_ratio_info"], value=0.5, interactive=True)
            with gr.Row():
                output_model = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
            with gr.Row():
                model_a.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model_a], outputs=[model_path_a])
                model_b.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model_b], outputs=[model_path_b])
            with gr.Row():
                fushion_button.click(
                    fn=fushion_model,
                    inputs=[
                        name_to_save, 
                        model_path_a, 
                        model_path_b, 
                        ratio
                    ],
                    outputs=[name_to_save, output_model],
                    api_name="fushion_model"
                )
                fushion_button.click(fn=lambda: visible(True), inputs=[], outputs=[output_model])  

        with gr.TabItem(translations["read_model"], visible=configs.get("read_tab", True)):
            gr.Markdown(translations["read_model_markdown"])
            with gr.Row():
                gr.Markdown(translations["read_model_markdown_2"])
            with gr.Row():
                model = gr.File(label=translations["drop_model"], file_types=[".pth", ".onnx"]) 
            with gr.Row():
                read_button = gr.Button(translations["readmodel"], variant="primary", scale=2)
            with gr.Column():
                model_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
                output_info = gr.Textbox(label=translations["modelinfo"], value="", interactive=False, scale=6)
            with gr.Row():
                model.upload(fn=lambda model: shutil.move(model.name, os.path.join("assets", "weights")), inputs=[model], outputs=[model_path])
                read_button.click(
                    fn=model_info,
                    inputs=[model_path],
                    outputs=[output_info],
                    api_name="read_model"
                )

        with gr.TabItem(translations["convert_model"], visible=configs.get("onnx_tab", True)):
            gr.Markdown(translations["pytorch2onnx"])
            with gr.Row():
                gr.Markdown(translations["pytorch2onnx_markdown"])
            with gr.Row():
                model_pth_upload = gr.File(label=translations["drop_model"], file_types=[".pth"]) 
            with gr.Row():
                convert_onnx = gr.Button(translations["convert_model"], variant="primary", scale=2)
            with gr.Row():
                model_pth_path = gr.Textbox(label=translations["model_path"], value="", placeholder="assets/weights/Model.pth", info=translations["model_path_info"], interactive=True)
            with gr.Row():
                output_model2 = gr.File(label=translations["output_model_path"], file_types=[".pth", ".onnx"], interactive=False, visible=False)
            with gr.Row():
                model_pth_upload.upload(fn=lambda model_pth_upload: shutil.move(model_pth_upload.name, os.path.join("assets", "weights")), inputs=[model_pth_upload], outputs=[model_pth_path])
                convert_onnx.click(
                    fn=onnx_export,
                    inputs=[model_pth_path],
                    outputs=[output_model2, output_info],
                    api_name="model_onnx_export"
                )
                convert_onnx.click(fn=lambda: visible(True), inputs=[], outputs=[output_model2])  

        with gr.TabItem(translations["f0_extractor_tab"], visible=configs.get("f0_extractor_tab", True)):
            gr.Markdown(translations["f0_extractor_markdown"])
            with gr.Row():
                gr.Markdown(translations["f0_extractor_markdown_2"])
            with gr.Row():
                extractor_button = gr.Button(translations["extract_button"].replace("2. ", ""), variant="primary")
            with gr.Row():
                with gr.Column():
                    upload_audio_file = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
                    audioplay = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
                with gr.Column():
                    with gr.Accordion(translations["f0_method"], open=False):
                        with gr.Group():
                            onnx_f0_mode3 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                            f0_method_extract = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                    with gr.Accordion(translations["audio_path"], open=True):
                        input_audio_path = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                        refesh_audio_button = gr.Button(translations["refesh"])
            with gr.Row():
                gr.Markdown("___")
            with gr.Row():
                file_output = gr.File(label="", file_types=[".txt"], interactive=False)
                image_output = gr.Image(label="", interactive=False, show_download_button=True)
            with gr.Row():
                upload_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio_file], outputs=[input_audio_path])
                input_audio_path.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio_path], outputs=[audioplay])
                refesh_audio_button.click(fn=change_audios_choices, inputs=[input_audio_path], outputs=[input_audio_path])
            with gr.Row():
                extractor_button.click(
                    fn=f0_extract,
                    inputs=[
                        input_audio_path,
                        f0_method_extract,
                        onnx_f0_mode3
                    ],
                    outputs=[file_output, image_output],
                    api_name="f0_extract"
                )

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
