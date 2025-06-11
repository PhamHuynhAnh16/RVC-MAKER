import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.editing import run_audioldm2
from main.app.core.utils import google_translate
from main.app.core.ui import change_audios_choices, shutil_move
from main.app.variables import translations, paths_for_files, sample_rate_choice, google_tts_voice, configs

def audio_editing_tab():
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
                        audioldm2_sample_rate = gr.Radio(choices=sample_rate_choice, label=translations["sr"], info=translations["sr_info"], value=44100, interactive=True)
                        t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label=translations["t_start"], interactive=True, info=translations["t_start_info"])
                        steps = gr.Slider(value=50, step=1, minimum=10, maximum=300, label=translations["steps_label"], info=translations["steps_info"], interactive=True)
            with gr.Accordion(translations["translate"], open=False):
                        with gr.Row():
                            source_lang2 = gr.Dropdown(label=translations["source_lang"], choices=["auto"]+google_tts_voice, interactive=True, value="auto")
                            target_lang2 = gr.Dropdown(label=translations["target_lang"], choices=google_tts_voice, interactive=True, value="en")
                        with gr.Row():
                            translate_button2 = gr.Button(" ".join([translations["translate"], translations["target_prompt"]]))
                            translate_button3 = gr.Button(" ".join([translations["translate"], translations["source_prompt"]]))
    with gr.Row():
        gr.Markdown(translations["output_audio"])
    with gr.Row():
        output_audioldm2 = gr.Audio(show_download_button=True, interactive=False, label=translations["output_audio"])
    with gr.Row():
        translate_button2.click(fn=google_translate, inputs=[tar_prompt, source_lang2, target_lang2], outputs=[tar_prompt], api_name="google_translate2")
        translate_button3.click(fn=google_translate, inputs=[src_prompt, source_lang2, target_lang2], outputs=[src_prompt], api_name="google_translate3")
    with gr.Row():
        refesh_audio.click(fn=change_audios_choices, inputs=[input_audiopath], outputs=[input_audiopath])
        drop_audio_file.upload(fn=lambda audio_in: shutil_move(audio_in.name, configs["audios_path"]), inputs=[drop_audio_file], outputs=[input_audiopath])
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