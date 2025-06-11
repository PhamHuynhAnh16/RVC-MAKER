import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import translations, configs
from main.app.tabs.extra.child.fushion import fushion_tab
from main.app.tabs.extra.child.settings import settings_tab
from main.app.tabs.extra.child.read_model import read_model_tab
from main.app.tabs.extra.child.f0_extract import f0_extract_tab
from main.app.tabs.extra.child.report_bugs import report_bugs_tab
from main.app.tabs.extra.child.convert_model import convert_model_tab

def extra_tab(app):
    with gr.TabItem(translations["extra"], visible=configs.get("extra_tab", True)):
        with gr.TabItem(translations["fushion"], visible=configs.get("fushion_tab", True)):
            gr.Markdown(translations["fushion_markdown"])
            fushion_tab()

        with gr.TabItem(translations["read_model"], visible=configs.get("read_tab", True)):
            gr.Markdown(translations["read_model_markdown"])
            read_model_tab()

        with gr.TabItem(translations["convert_model"], visible=configs.get("onnx_tab", True)):
            gr.Markdown(translations["pytorch2onnx"])
            convert_model_tab()

        with gr.TabItem(translations["f0_extractor_tab"], visible=configs.get("f0_extractor_tab", True)):
            gr.Markdown(translations["f0_extractor_markdown"])
            f0_extract_tab()

        with gr.TabItem(translations["settings"], visible=configs.get("settings_tab", True)):
            gr.Markdown(translations["settings_markdown"])
            settings_tab(app)

        
            report_bugs_tab()
