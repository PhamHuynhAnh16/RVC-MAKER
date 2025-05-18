from main.tools import huggingface
from main.configs.config import Config
from main.app.based.utils import *
import gradio as gr


def model_tabs():
    with gr.Tabs():
        with gr.Tab(label=translations["downloads"], visible=configs.get("downloads_tab", True)):
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

        with gr.Tab(label=translations["createdataset"], visible=configs.get("create_dataset_tab", True)):
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

        with gr.Tab(label=translations["training_model"], visible=configs.get("training_tab", True)):
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

        with gr.Tab(label=translations["fushion"], visible=configs.get("fushion_tab", True)):
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

        with gr.Tab(label=translations["read_model"], visible=configs.get("read_tab", True)):
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

        with gr.Tab(label=translations["convert_model"], visible=configs.get("onnx_tab", True)):
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