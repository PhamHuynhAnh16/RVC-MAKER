import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.variables import configs, translations
from main.app.tabs.editing.child.quirk import quirk_tab
from main.app.tabs.editing.child.audio_editing import audio_editing_tab
from main.app.tabs.editing.child.audio_effects import audio_effects_tab

def editing_tab():
    with gr.TabItem(translations["editing"], visible=configs.get("editing_tab", True)):
        with gr.TabItem(translations["audio_effects"], visible=configs.get("effects_tab", True)):
            gr.Markdown(translations["apply_audio_effects"])
            audio_effects_tab()
            
        with gr.TabItem(translations["quirk"], visible=configs.get("quirk", True)):
            gr.Markdown(translations["quirk_info"])
            quirk_tab()

        with gr.TabItem(translations["audio_editing"], visible=configs.get("audioldm2", True)):
            gr.Markdown(translations["audio_editing_info"])
            audio_editing_tab()
            