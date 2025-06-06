import gradio as gr
from main.tools import huggingface
from main.configs.config import Config
from main.app.based.utils import *

def utils_tabs():
    with gr.TabItem("utils"):
        with gr.Tabs():
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
