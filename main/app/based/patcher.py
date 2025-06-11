import gradio as gr
import logging, torch

def gr_info(message):
    gr.Info(message, duration=2)
    logger.info(message)

def gr_warning(message):
    gr.Warning(message, duration=2)
    logger.warning(message)

def gr_error(message):
    gr.Error(message=message, duration=6)
    logger.error(message)

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = [f"{i}: {torch.cuda.get_device_name(i)} ({int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4)} GB)" for i in range(ngpu) if torch.cuda.is_available() or ngpu != 0]
    return "\n".join(gpu_infos) if len(gpu_infos) > 0 else translations["no_support_gpu"]
