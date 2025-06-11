import gradio as gr
import os
import shutil
from main.tools import Config
from main.tools.utils import *
from main.tools.huggingface import download_url, separator_music

def update_visibility(backing, reverb, separator_model, cleaner):
    is_mdx = separator_model in mdx_model
    return {
        "mdx_batch_size": gr.update(visible=backing or reverb or is_mdx),
        "mdx_hop_length": gr.update(visible=backing or reverb or is_mdx),
        "denoise": gr.update(visible=is_mdx, interactive=is_mdx),
        "shifts": gr.update(visible=not is_mdx),
        "separator_backing_model": gr.update(visible=backing),
        "main_vocals": gr.update(visible=backing),
        "backing_vocals": gr.update(visible=backing),
        "backing_reverb": gr.update(visible=backing and reverb, interactive=backing and reverb),
        "clean_strength": gr.update(visible=cleaner)
    }

def uvr_tabs():
    with gr.TabItem(translations["separator_tab"], visible=configs.get("separator_tab", True)):
        gr.Markdown(f"# {translations['separator_tab']}")
        
        # Input Section
        with gr.Group():
            gr.Markdown("### Upload Audio")
            with gr.Row():
                input_file = gr.File(
                    label=translations["drop_audio"],
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"]
                )
                audio_preview = gr.Audio(
                    label=translations["input_audio"],
                    show_download_button=True,
                    interactive=False
                )
            with gr.Accordion(translations["use_url"], open=False):
                url = gr.Textbox(
                    label=translations["url_audio"],
                    placeholder="https://www.youtube.com/...",
                    value=""
                )
                download_button = gr.Button(translations["downloads"])

        # Basic Settings
        with gr.Group():
            gr.Markdown("### Basic Settings")
            with gr.Row():
                separator_model = gr.Dropdown(
                    label=translations["separator_model"],
                    value=uvr_model[0],
                    choices=uvr_model,
                    interactive=True
                )
                output_format = gr.Radio(
                    label=translations["export_format"],
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                    interactive=True
                )
            with gr.Row():
                input_audio = gr.Dropdown(
                    label=translations["audio_path"],
                    value="",
                    choices=paths_for_files,
                    allow_custom_value=True,
                    interactive=True
                )
                refresh_button = gr.Button(translations["refresh"])
            output_folder = gr.Textbox(
                label=translations["output_folder"],
                value="audios",
                placeholder="audios",
                interactive=True
            )

        # Processing Options
        with gr.Accordion("Advanced Processing Options", open=False):
            with gr.Group():
                gr.Markdown("### Processing Options")
                with gr.Row():
                    cleaner = gr.Checkbox(
                        label=translations["clear_audio"],
                        value=False,
                        interactive=True
                    )
                    backing = gr.Checkbox(
                        label=translations["separator_backing"],
                        value=False,
                        interactive=True
                    )
                    reverb = gr.Checkbox(
                        label=translations["dereverb_audio"],
                        value=False,
                        interactive=True
                    )
                with gr.Row():
                    clean_strength = gr.Slider(
                        label=translations["clean_strength"],
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.1,
                        interactive=True,
                        visible=False
                    )
                    separator_backing_model = gr.Dropdown(
                        label=translations["separator_backing_model"],
                        value="Version-1",
                        choices=["Version-1", "Version-2"],
                        interactive=True,
                        visible=False
                    )
                with gr.Row():
                    backing_reverb = gr.Checkbox(
                        label=translations["dereverb_backing"],
                        value=False,
                        interactive=False,
                        visible=False
                    )
                    denoise = gr.Checkbox(
                        label=translations["denoise_mdx"],
                        value=False,
                        interactive=False,
                        visible=False
                    )

            # Fine-tuning Parameters
            with gr.Group():
                gr.Markdown("### Fine-tuning Parameters")
                with gr.Row():
                    shifts = gr.Slider(
                        label=translations["shift"],
                        minimum=1,
                        maximum=20,
                        value=2,
                        step=1,
                        interactive=True
                    )
                    segment_size = gr.Slider(
                        label=translations["segments_size"],
                        minimum=32,
                        maximum=3072,
                        value=256,
                        step=32,
                        interactive=True
                    )
                with gr.Row():
                    overlap = gr.Radio(
                        label=translations["overlap"],
                        choices=["0.25", "0.5", "0.75"],
                        value="0.25",
                        interactive=True
                    )
                    sample_rate = gr.Slider(
                        minimum=8000,
                        maximum=96000,
                        step=1,
                        value=44100,
                        label=translations["sr"],
                        interactive=True
                    )
                with gr.Row():
                    mdx_batch_size = gr.Slider(
                        label=translations["batch_size"],
                        minimum=1,
                        maximum=64,
                        value=1,
                        step=1,
                        interactive=True,
                        visible=False
                    )
                    mdx_hop_length = gr.Slider(
                        label="Hop length",
                        minimum=1,
                        maximum=8192,
                        value=1024,
                        step=1,
                        interactive=True,
                        visible=False
                    )

        # Process Button
        separator_button = gr.Button(translations["separator_tab"], variant="primary")

        # Output Section
        with gr.Group():
            gr.Markdown("### Output Audio")
            with gr.Row():
                instruments_audio = gr.Audio(
                    label=translations["instruments"],
                    show_download_button=True,
                    interactive=False
                )
                original_vocals = gr.Audio(
                    label=translations["original_vocal"],
                    show_download_button=True,
                    interactive=False
                )
            with gr.Row():
                main_vocals = gr.Audio(
                    label=translations["main_vocal"],
                    show_download_button=True,
                    interactive=False,
                    visible=False
                )
                backing_vocals = gr.Audio(
                    label=translations["backing_vocal"],
                    show_download_button=True,
                    interactive=False,
                    visible=False
                )

        # Event Handlers
        separator_model.change(
            fn=update_visibility,
            inputs=[backing, reverb, separator_model, cleaner],
            outputs=[
                mdx_batch_size, mdx_hop_length, denoise, shifts,
                separator_backing_model, main_vocals, backing_vocals,
                backing_reverb, clean_strength
            ]
        )
        backing.change(
            fn=update_visibility,
            inputs=[backing, reverb, separator_model, cleaner],
            outputs=[
                mdx_batch_size, mdx_hop_length, denoise, shifts,
                separator_backing_model, main_vocals, backing_vocals,
                backing_reverb, clean_strength
            ]
        )
        reverb.change(
            fn=update_visibility,
            inputs=[backing, reverb, separator_model, cleaner],
            outputs=[
                mdx_batch_size, mdx_hop_length, denoise, shifts,
                separator_backing_model, main_vocals, backing_vocals,
                backing_reverb, clean_strength
            ]
        )
        cleaner.change(
            fn=update_visibility,
            inputs=[backing, reverb, separator_model, cleaner],
            outputs=[
                mdx_batch_size, mdx_hop_length, denoise, shifts,
                separator_backing_model, main_vocals, backing_vocals,
                backing_reverb, clean_strength
            ]
        )
        input_audio.change(
            fn=lambda audio: audio if os.path.isfile(audio) else None,
            inputs=[input_audio],
            outputs=[audio_preview]
        )
        input_file.upload(
            fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")),
            inputs=[input_file],
            outputs=[input_audio]
        )
        refresh_button.click(
            fn=change_audios_choices,
            inputs=[input_audio],
            outputs=[input_audio]
        )
        download_button.click(
            fn=download_url,
            inputs=[url],
            outputs=[input_audio, audio_preview, url],
            api_name='download_url'
        )
        separator_button.click(
            fn=separator_music,
            inputs=[
                input_audio, output_folder, output_format, shifts,
                segment_size, overlap, cleaner, clean_strength, denoise,
                separator_model, separator_backing_model, backing, reverb,
                backing_reverb, mdx_hop_length, mdx_batch_size, sample_rate
            ],
            outputs=[original_vocals, instruments_audio, main_vocals, backing_vocals],
            api_name='separator_music'
        )
