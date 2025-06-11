import gradio as gr
from main.tools import huggingface
from main.configs.config import Config
from main.app.based.utils import *



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
                refresh_button = gr.Button("refresh")
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

def inference_tabs():
    # Audio Conversion Tab
    with gr.TabItem(translations["convert_audio"], visible=configs.get("convert_tab", True)):
        gr.Markdown(f"## {translations['convert_audio']}")
        with gr.Row():
            gr.Markdown(translations["convert_info"])
        
        with gr.Row():
            with gr.Column():
                with gr.Accordion(translations["model_accordion"], open=True):
                    with gr.Row(equal_height=True):
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

    # Text-to-Speech Conversion Tab
    with gr.TabItem(translations["convert_text"], visible=configs.get("tts_tab", True)):
        gr.Markdown(translations["convert_text_markdown"])
        with gr.Row():
            gr.Markdown(translations["convert_text_markdown_2"])
        with gr.Accordion(translations["model_accordion"], open=True):
            with gr.Row(equal_height=True):
                model_pth0 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                model_index0 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                refesh1 = gr.Button(translations["refesh"])
                    
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

    # Whisper Conversion Tab
    with gr.TabItem(translations["convert_with_whisper"], visible=configs.get("convert_with_whisper", True)):
        gr.Markdown(f"## {translations['convert_with_whisper']}")
        with gr.Row():
            gr.Markdown(translations["convert_with_whisper_info"])
        with gr.Row():
            with gr.Column():
                with gr.Accordion(translations["model_accordion"] + " 1", open=True):
                    with gr.Row(equal_height=True):
                        model_pth2 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                        model_index2 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        refesh2 = gr.Button(translations["refesh"])
                with gr.Accordion(translations["model_accordion"] + " 2", open=True):
                    with gr.Row(equal_height=True):
                        model_pth3 = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                        model_index3 = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        refesh3 = gr.Button(translations["refesh"])
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
