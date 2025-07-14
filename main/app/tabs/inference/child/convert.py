import os
import sys
import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.presets import load_presets, save_presets
from main.app.core.inference.inference import convert_audio, convert_selection
from main.app.variables import (
    translations, paths_for_files, sample_rate_choice, model_name, index_path,
    method_f0, f0_file, embedders_mode, embedders_model, presets_file, configs
)
from main.app.core.ui import (
    visible, valueFalse_interactive, change_audios_choices, change_f0_choices,
    unlock_f0, change_preset_choices, change_backing_choices, hoplength_show,
    change_models_choices, get_index, index_strength_show, visible_embedders, shutil_move
)

def convert_tab():
    # Model Selection Section
    with gr.Row(equal_height=True):
        model_pth = gr.Dropdown(
            label=translations["model_name"],
            choices=model_name,
            value=model_name[0] if model_name else "",
            interactive=True,
            allow_custom_value=True
        )
        model_index = gr.Dropdown(
            label=translations["index_path"],
            choices=index_path,
            value=index_path[0] if index_path else "",
            interactive=True,
            allow_custom_value=True
        )
        with gr.Row(equal_height=True):
            refresh_models = gr.Button(translations["refesh"])
    # Conversion Settings Section
    with gr.Row(equal_height=True):
        with gr.Column():
            pitch = gr.Slider(
                minimum=-20,
                maximum=20,
                step=1,
                info=translations["pitch_info"],
                label=translations["pitch"],
                value=0,
                interactive=True
            )
            clean_strength = gr.Slider(
                label=translations["clean_strength"],
                info=translations["clean_strength_info"],
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.1,
                interactive=True,
                visible=False
            )
            index_strength = gr.Slider(
                label=translations["index_strength"],
                info=translations["index_strength_info"],
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01,
                interactive=True,
                visible=bool(model_index.value)
            )

    # Audio Input Section
    with gr.Row():
        with gr.Column():
            input_audio = gr.File(
                label=translations["drop_audio"],
                file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"]
            )
            play_audio = gr.Audio(
                show_download_button=True,
                interactive=False,
                label=translations["input_audio"]
            )
            input_audio_path = gr.Dropdown(
                label=translations["audio_path"],
                value="",
                choices=paths_for_files,
                info=translations["provide_audio"],
                allow_custom_value=True,
                interactive=True
            )
            refresh_audio = gr.Button(translations["refesh"])

    
    # Advanced Settings Accordion
    with gr.Accordion(translations["setting"], open=False):
        with gr.Group():
            with gr.Row():
                cleaner = gr.Checkbox(label=translations["clear_audio"], value=False, interactive=True)
                autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                use_audio = gr.Checkbox(label=translations["use_audio"], value=False, interactive=True)
                checkpointing = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
            with gr.Row():
                use_original = gr.Checkbox(label=translations["convert_original"], value=False, interactive=True, visible=False)
                convert_backing = gr.Checkbox(label=translations["convert_backing"], value=False, interactive=True, visible=False)
                not_merge_backing = gr.Checkbox(label=translations["not_merge_backing"], value=False, interactive=True, visible=False)
                merge_instrument = gr.Checkbox(label=translations["merge_instruments"], value=False, interactive=True, visible=False)
            with gr.Row():
                split_audio = gr.Checkbox(label=translations["split_audio"], value=False, interactive=True)
                formant_shifting = gr.Checkbox(label=translations["formantshift"], value=False, interactive=True)
                auto_pitch = gr.Checkbox(label=translations["proposal_pitch"], value=False, interactive=True)
            with gr.Row():
                resample_sr = gr.Radio(
                    choices=[0] + sample_rate_choice,
                    label=translations["resample"],
                    info=translations["resample_info"],
                    value=0,
                    interactive=True
                )
                f0_autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=translations["autotune_rate"],
                    info=translations["autotune_rate_info"],
                    value=1,
                    step=0.1,
                    interactive=True,
                    visible=False
                )
                # proposal_pitch_threshold = gr.Slider(
                #     minimum=50.0,
                #     maximum=1200.0,
                #     label=translations["proposal_pitch_threshold"], 
                #     info=translations["proposal_pitch_threshold_info"], 
                #     value=255.0, 
                #     step=0.1, 
                #     interactive=True, 
                #     visible=proposal_pitch.value
                # )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=translations["filter_radius"],
                    info=translations["filter_radius_info"],
                    value=3,
                    step=1,
                    interactive=True
                )
                volume_envelope = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=translations["volume_envelope"],
                    info=translations["volume_envelope_info"],
                    value=1,
                    step=0.1,
                    interactive=True
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=translations["protect"],
                    info=translations["protect_info"],
                    value=0.5,
                    step=0.01,
                    interactive=True
                )
                formant_qfrency = gr.Slider(
                    value=1.0,
                    label=translations["formant_qfrency"],
                    info=translations["formant_qfrency"],
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    interactive=True,
                    visible=False
                )
                formant_timbre = gr.Slider(
                    value=1.0,
                    label=translations["formant_timbre"],
                    info=translations["formant_timbre"],
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    interactive=True,
                    visible=False
                )

    # F0 Method Settings
    with gr.Accordion(translations["f0_method"], open=False):
        with gr.Group():
            with gr.Row():
                onnx_f0_mode = gr.Checkbox(
                    label=translations["f0_onnx_mode"],
                    info=translations["f0_onnx_mode_info"],
                    value=False,
                    interactive=True
                )
                unlock_full_method = gr.Checkbox(
                    label=translations["f0_unlock"],
                    info=translations["f0_unlock_info"],
                    value=False,
                    interactive=True
                )
            method = gr.Radio(
                label=translations["f0_method"],
                info=translations["f0_method_info"],
                choices=method_f0,
                value="rmvpe",
                interactive=True
            )
            hybrid_method = gr.Dropdown(
                label=translations["f0_method_hybrid"],
                info=translations["f0_method_hybrid_info"],
                choices=[
                    "hybrid[pm+dio]", "hybrid[pm+crepe-tiny]", "hybrid[pm+crepe]",
                    "hybrid[pm+fcpe]", "hybrid[pm+rmvpe]", "hybrid[pm+harvest]",
                    "hybrid[pm+yin]", "hybrid[dio+crepe-tiny]", "hybrid[dio+crepe]",
                    "hybrid[dio+fcpe]", "hybrid[dio+rmvpe]", "hybrid[dio+harvest]",
                    "hybrid[dio+yin]", "hybrid[crepe-tiny+crepe]", "hybrid[crepe-tiny+fcpe]",
                    "hybrid[crepe-tiny+rmvpe]", "hybrid[crepe-tiny+harvest]",
                    "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe]", "hybrid[crepe+harvest]",
                    "hybrid[crepe+yin]", "hybrid[fcpe+rmvpe]", "hybrid[fcpe+harvest]",
                    "hybrid[fcpe+yin]", "hybrid[rmvpe+harvest]", "hybrid[rmvpe+yin]",
                    "hybrid[harvest+yin]"
                ],
                value="hybrid[pm+dio]",
                interactive=True,
                allow_custom_value=True,
                visible=False
            )
            hop_length = gr.Slider(
                label="Hop length",
                info=translations["hop_length_info"],
                minimum=1,
                maximum=512,
                value=128,
                step=1,
                interactive=True,
                visible=False
            )

    # F0 File Settings
    with gr.Accordion(translations["f0_file"], open=False):
        upload_f0_file = gr.File(label=translations["upload_f0"], file_types=[".txt"])
        f0_file_dropdown = gr.Dropdown(
            label=translations["f0_file_2"],
            value="",
            choices=f0_file,
            allow_custom_value=True,
            interactive=True
        )
        refresh_f0_file = gr.Button(translations["refesh"])

    # Hubert Model Settings
    with gr.Accordion(translations["hubert_model"], open=False):
        embed_mode = gr.Radio(
            label=translations["embed_mode"],
            info=translations["embed_mode_info"],
            value="fairseq",
            choices=embedders_mode,
            interactive=True
        )
        embedders = gr.Radio(
            label=translations["hubert_model"],
            info=translations["hubert_info"],
            choices=embedders_model,
            value="hubert_base",
            interactive=True
        )
        custom_embedders = gr.Textbox(
            label=translations["modelname"],
            info=translations["modelname_info"],
            value="",
            placeholder="hubert_base",
            interactive=True,
            visible=False
        )

    # Presets Section
    with gr.Accordion(translations["use_presets"], open=False):
        presets_name = gr.Dropdown(
            label=translations["file_preset"],
            choices=presets_file,
            value=presets_file[0] if presets_file else "",
            interactive=True,
            allow_custom_value=True
        )
        with gr.Row():
            load_presets_btn = gr.Button(translations["load_file"], variant="primary")
            refresh_presets = gr.Button(translations["refesh"])
        with gr.Accordion(translations["export_file"], open=False):
            with gr.Group():
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
            name_to_save_file = gr.Textbox(label=translations["filename_to_save"])
            save_file_button = gr.Button(translations["export_file"])
        upload_presets = gr.File(label=translations["upload_presets"], file_types=[".conversion.json"])

    # Output Settings
    with gr.Accordion(translations["input_output"], open=False):
        export_format = gr.Radio(
            label=translations["export_format"],
            info=translations["export_info"],
            choices=["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"],
            value="wav",
            interactive=True
        )
        output_audio_path = gr.Textbox(
            label=translations["output_path"],
            value="audios/output.wav",
            placeholder="audios/output.wav",
            info=translations["output_path_info"],
            interactive=True
        )

    # Conversion Buttons
    with gr.Row():
        convert_button = gr.Button(translations["convert_audio"], variant="primary")
        audio_select = gr.Dropdown(
            label=translations["select_separate"],
            choices=[],
            value="",
            interactive=True,
            allow_custom_value=True,
            visible=False
        )
        convert_button_2 = gr.Button(translations["convert_audio"], visible=False)

    # Output Audio Section
    with gr.Row():
        gr.Markdown(translations["output_convert"])
    with gr.Row():
        main_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["main_convert"])
        backing_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_backing"], visible=False)
        main_backing = gr.Audio(show_download_button=True, interactive=False, label=translations["main_or_backing"], visible=False)
        original_convert = gr.Audio(show_download_button=True, interactive=False, label=translations["convert_original"], visible=False)
        vocal_instrument = gr.Audio(show_download_button=True, interactive=False, label=translations["voice_or_instruments"], visible=False)

    # Event Handlers
    refresh_models.click(fn=change_models_choices, inputs=[], outputs=[model_pth, model_index])
    model_pth.change(fn=get_index, inputs=[model_pth], outputs=[model_index])
    model_index.change(fn=index_strength_show, inputs=[model_index], outputs=[index_strength])

    input_audio.upload(fn=lambda audio_in: shutil_move(audio_in.name, configs["audios_path"]), inputs=[input_audio], outputs=[input_audio_path])
    input_audio_path.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio_path], outputs=[play_audio])
    refresh_audio.click(fn=change_audios_choices, inputs=[input_audio_path], outputs=[input_audio_path])

    cleaner.change(fn=visible, inputs=[cleaner], outputs=[clean_strength])
    autotune.change(fn=visible, inputs=[autotune], outputs=[f0_autotune_strength])
    formant_shifting.change(fn=lambda a: [visible(a)]*2, inputs=[formant_shifting], outputs=[formant_qfrency, formant_timbre])

    use_audio.change(
        fn=lambda a: [visible(a)]*4 + [valueFalse_interactive(a)]*4 + [visible(not a)]*4,
        inputs=[use_audio],
        outputs=[
            main_backing, use_original, convert_backing, not_merge_backing, merge_instrument,
            use_original, convert_backing, not_merge_backing, merge_instrument,
            input_audio_path, output_audio_path, input_audio, play_audio
        ]
    )
    convert_backing.change(
        fn=lambda a, b: [change_backing_choices(a, b), visible(a)],
        inputs=[convert_backing, not_merge_backing],
        outputs=[use_original, backing_convert]
    )
    use_original.change(
        fn=lambda a, b: [visible(b), visible(not b), visible(a and not b), valueFalse_interactive(not b), valueFalse_interactive(not b)],
        inputs=[use_audio, use_original],
        outputs=[original_convert, main_convert, main_backing, convert_backing, not_merge_backing]
    )
    not_merge_backing.change(
        fn=lambda a, b, c: [visible(a and not b), change_backing_choices(c, b)],
        inputs=[use_audio, not_merge_backing, convert_backing],
        outputs=[main_backing, use_original]
    )
    merge_instrument.change(fn=visible, inputs=[merge_instrument], outputs=[vocal_instrument])

    method.change(
        fn=lambda m, h: [visible(m == "hybrid"), hoplength_show(m, h)],
        inputs=[method, hybrid_method],
        outputs=[hybrid_method, hop_length]
    )
    hybrid_method.change(fn=hoplength_show, inputs=[method, hybrid_method], outputs=[hop_length])
    unlock_full_method.change(fn=unlock_f0, inputs=[unlock_full_method], outputs=[method])

    upload_f0_file.upload(fn=lambda inp: shutil_move(inp.name, configs["f0_path"]), inputs=[upload_f0_file], outputs=[f0_file_dropdown])
    refresh_f0_file.click(fn=change_f0_choices, inputs=[], outputs=[f0_file_dropdown])

    embed_mode.change(fn=visible_embedders, inputs=[embed_mode], outputs=[embedders])
    embedders.change(fn=lambda e: visible(e == "custom"), inputs=[embedders], outputs=[custom_embedders])

    load_presets_btn.click(
        fn=load_presets,
        inputs=[
            presets_name, cleaner, autotune, pitch, clean_strength, index_strength,
            resample_sr, filter_radius, volume_envelope, protect, split_audio,
            f0_autotune_strength, formant_qfrency, formant_timbre
        ],
        outputs=[
            cleaner, autotune, pitch, clean_strength, index_strength, resample_sr,
            filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength,
            formant_shifting, formant_qfrency, formant_timbre
        ]
    )
    refresh_presets.click(fn=change_preset_choices, inputs=[], outputs=[presets_name])
    save_file_button.click(
        fn=save_presets,
        inputs=[
            name_to_save_file, cleaner, autotune, pitch, clean_strength, index_strength,
            resample_sr, filter_radius, volume_envelope, protect, split_audio,
            f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox,
            index_strength_chbox, resample_sr_chbox, filter_radius_chbox,
            volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox,
            formant_shifting, formant_qfrency, formant_timbre
        ],
        outputs=[presets_name]
    )
    upload_presets.upload(fn=lambda audio_in: shutil_move(audio_in.name, configs["presets_path"]), inputs=[upload_presets], outputs=[presets_name])
    #proposal_pitch.change(fn=visible, inputs=[proposal_pitch], outputs=[proposal_pitch_threshold])

    audio_select.change(fn=lambda: visible(True), inputs=[], outputs=[convert_button_2])
    convert_button.click(fn=lambda: visible(False), inputs=[], outputs=[convert_button])
    convert_button_2.click(fn=lambda: [visible(False), visible(False)], inputs=[], outputs=[audio_select, convert_button_2])

    convert_button.click(
        fn=convert_selection,
        inputs=[
            cleaner, autotune, use_audio, use_original, convert_backing, not_merge_backing,
            merge_instrument, pitch, clean_strength, model_pth, model_index, index_strength,
            input_audio_path, output_audio_path, export_format, method, hybrid_method,
            hop_length, embedders, custom_embedders, resample_sr, filter_radius,
            volume_envelope, protect, split_audio, f0_autotune_strength, checkpointing,
            onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre, f0_file_dropdown,
            embed_mode, auto_pitch, #proposal_pitch_threshold
        ],
        outputs=[audio_select, main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
        api_name="convert_selection"
    )
    convert_button_2.click(
        fn=convert_audio,
        inputs=[
            cleaner, autotune, use_audio, use_original, convert_backing, not_merge_backing,
            merge_instrument, pitch, clean_strength, model_pth, model_index, index_strength,
            input_audio_path, output_audio_path, export_format, method, hybrid_method,
            hop_length, embedders, custom_embedders, resample_sr, filter_radius,
            volume_envelope, protect, split_audio, f0_autotune_strength, audio_select,
            checkpointing, onnx_f0_mode, formant_shifting, formant_qfrency, formant_timbre,
            f0_file_dropdown, embed_mode, auto_pitch, #proposal_pitch_threshold
        ],
        outputs=[main_convert, backing_convert, main_backing, original_convert, vocal_instrument, convert_button],
        api_name="convert_audio"
    )

    return convert_tab
