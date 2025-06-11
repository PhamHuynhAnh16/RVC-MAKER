import os
import sys
import codecs

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.utils import report_bug
from main.app.variables import translations

def report_bugs_tab():
    with gr.Row():
        gr.Markdown(translations["report_bug_info"])
    with gr.Row():
        with gr.Column():
            with gr.Group():
                agree_log = gr.Checkbox(label=translations["agree_log"], value=True, interactive=True) 
                report_text = gr.Textbox(label=translations["error_info"], info=translations["error_info_2"], interactive=True)
            report_button = gr.Button(translations["report_bugs"], variant="primary", scale=2)
    with gr.Row():
        gr.Markdown(translations["report_info"].format(github=codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/vffhrf", "rot13")))
    with gr.Row():
        report_button.click(fn=report_bug, inputs=[report_text, agree_log], outputs=[])