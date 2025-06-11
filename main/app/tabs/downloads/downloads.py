import os
import sys
import yt_dlp
import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import translations, configs, models, model_options
from main.app.core.downloads import download_model, search_models, download_pretrained_model
from main.app.core.ui import change_download_choices, change_download_pretrained_choices, shutil_move
from main.app.core.process import fetch_pretrained_data, save_drop_model, update_sample_rate_dropdown

output_dir = os.path.join(os.getcwd(), 'audios')



def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl=f'{output_dir}/%(title)s.mp3')

    return download_path

def download_tab():
    with gr.TabItem("yt_dlp"):
        ytaudio = gr.Audio(label="Output")
        url_inputyt = gr.Textbox(label="URL")
        
        downloadyt = gr.Button(translations["downloads"], variant="primary")
        downloadyt.click(fn=yt_download, inputs=url_inputyt, outputs=ytaudio)
    
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
                fn=lambda pretrain_upload_g: shutil_move(pretrain_upload_g.name, configs["pretrained_custom_path"]), 
                inputs=[pretrain_upload_g], 
                outputs=[],
                api_name="upload_pretrain_g"
            )
            pretrain_upload_d.upload(
                fn=lambda pretrain_upload_d: shutil_move(pretrain_upload_d.name, configs["pretrained_custom_path"]), 
                inputs=[pretrain_upload_d], 
                outputs=[],
                api_name="upload_pretrain_d"
            )
