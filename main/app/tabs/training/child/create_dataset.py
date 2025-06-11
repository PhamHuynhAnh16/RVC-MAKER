import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.training import create_dataset
from main.app.core.ui import visible, valueEmpty_visible1
from main.app.variables import translations, sample_rate_choice

def create_dataset_tab():
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
                    sample_rate0 = gr.Radio(choices=sample_rate_choice, value=44100, label=translations["sr"], info=translations["sr_info"], interactive=True)
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