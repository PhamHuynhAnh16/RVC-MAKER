import os
import sys
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import python, translations, configs

def separator_music(input, output_audio, format, shifts, segments_size, overlap, clean_audio, clean_strength, denoise, separator_model, kara_model, backing, reverb, backing_reverb, hop_length, batch_size, sample_rate):
    output = os.path.dirname(output_audio) or output_audio

    if not input or not os.path.exists(input) or os.path.isdir(input): 
        gr_warning(translations["input_not_valid"])
        return [None]*4
    
    if not os.path.exists(output): 
        gr_warning(translations["output_not_valid"])
        return [None]*4

    if not os.path.exists(output): os.makedirs(output)
    gr_info(translations["start"].format(start=translations["separator_music"]))

    subprocess.run([python, configs["separate_path"], "--input_path", input, "--output_path", output, "--format", format, "--shifts", str(shifts), "--segments_size", str(segments_size), "--overlap", str(overlap), "--mdx_hop_length", str(hop_length), "--mdx_batch_size", str(batch_size), "--clean_audio", str(clean_audio), "--clean_strength", str(clean_strength), "--kara_model", kara_model, "--backing", str(backing), "--mdx_denoise", str(denoise), "--reverb", str(reverb), "--backing_reverb", str(backing_reverb), "--model_name", separator_model, "--sample_rate", str(sample_rate)])
    gr_info(translations["success"])

    filename, _ = os.path.splitext(os.path.basename(input))
    output = os.path.join(output, filename)

    return [os.path.join(output, f"Original_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Original_Vocals.{format}"), os.path.join(output, f"Instruments.{format}"), (os.path.join(output, f"Main_Vocals_No_Reverb.{format}") if reverb else os.path.join(output, f"Main_Vocals.{format}") if backing else None), (os.path.join(output, f"Backing_Vocals_No_Reverb.{format}") if backing_reverb else os.path.join(output, f"Backing_Vocals.{format}") if backing else None)] if os.path.isfile(input) else [None]*4