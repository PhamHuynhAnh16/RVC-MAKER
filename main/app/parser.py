import os
import sys
import argparse
from typing import List, Dict, Callable
from dataclasses import dataclass

# Add current working directory to system path
sys.path.append(os.getcwd())

# Set environment variables for PyTorch
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

@dataclass
class Command:
    name: str
    module: str
    description: str
    help_text: str = ""

# Define available commands and their corresponding modules
COMMANDS: List[Command] = [
    Command("--audio_effects", "main.inference.audio_effects", "Apply audio effects"),
    Command("--audioldm2", "main.inference.audioldm2", "Process audio with AudioLDM2 model"),
    Command("--convert", "main.inference.conversion.convert", "Convert audio files"),
    Command("--create_dataset", "main.inference.create_dataset", "Create a training dataset"),
    Command("--create_index", "main.inference.create_index", "Create an index for a model"),
    Command("--extract", "main.inference.extract", "Extract features for training"),
    Command("--preprocess", "main.inference.preprocess.preprocess", "Preprocess audio data"),
    Command("--separator_music", "main.inference.separator_music", "Separate music components"),
    Command("--train", "main.inference.training.train", "Train a model"),
]

# Define help commands with their descriptions
HELP_COMMANDS: Dict[str, str] = {
    "--help_audio_effects": """
    Parameters for `--audio_effects`:
        1. File Paths:
            - `--input_path` (required): Path to the input audio file.
            - `--output_path` (default: `./audios/apply_effects.wav`): Path to save the output file.
            - `--export_format` (default: `wav`): Output file format (`wav`, `mp3`, etc.).

        2. Resampling:
            - `--resample` (default: `False`): Enable/disable resampling.
            - `--resample_sr` (default: `0`): New sampling rate (Hz).

        3. Chorus Effect:
            - `--chorus`: Enable/disable chorus effect.
            - `--chorus_depth`, `--chorus_rate`, `--chorus_mix`, `--chorus_delay`, `--chorus_feedback`: Chorus effect parameters.

        4. Distortion Effect:
            - `--distortion`: Enable/disable distortion effect.
            - `--drive_db`: Distortion intensity level.

        5. Reverb Effect:
            - `--reverb`: Enable/disable reverb effect.
            - `--reverb_room_size`, `--reverb_damping`, `--reverb_wet_level`, `--reverb_dry_level`, `--reverb_width`, `--reverb_freeze_mode`: Reverb effect parameters.

        6. Pitch Shift Effect:
            - `--pitchshift`: Enable/disable pitch shift effect.
            - `--pitch_shift`: Pitch shift value.

        7. Delay Effect:
            - `--delay`: Enable/disable delay effect.
            - `--delay_seconds`, `--delay_feedback`, `--delay_mix`: Delay effect parameters.

        8. Compressor:
            - `--compressor`: Enable/disable compressor effect.
            - `--compressor_threshold`, `--compressor_ratio`, `--compressor_attack_ms`, `--compressor_release_ms`: Compressor parameters.

        9. Limiter:
            - `--limiter`: Enable/disable limiter effect.
            - `--limiter_threshold`, `--limiter_release`: Limiter threshold and release time.

        10. Gain:
            - `--gain`: Enable/disable gain effect.
            - `--gain_db`: Gain level (dB).

        11. Bitcrush:
            - `--bitcrush`: Enable/disable bitcrush effect.
            - `--bitcrush_bit_depth`: Bit depth for bitcrush effect.

        12. Clipping:
            - `--clipping`: Enable/disable clipping effect.
            - `--clipping_threshold`: Clipping threshold.

        13. Phaser:
            - `--phaser`: Enable/disable phaser effect.
            - `--phaser_rate_hz`, `--phaser_depth`, `--phaser_centre_frequency_hz`, `--phaser_feedback`, `--phaser_mix`: Phaser effect parameters.

        14. Bass & Treble Boost:
            - `--treble_bass_boost`: Enable/disable bass and treble boost.
            - `--bass_boost_db`, `--bass_boost_frequency`, `--treble_boost_db`, `--treble_boost_frequency`: Bass and treble boost parameters.

        15. Fade In/Out:
            - `--fade_in_out`: Enable/disable fade effect.
            - `--fade_in_duration`, `--fade_out_duration`: Fade in/out durations.

        16. Audio Combination:
            - `--audio_combination`: Enable/disable combining multiple audio files.
            - `--audio_combination_input`: Path to additional audio file.
            - `--main_volume`: Volume of the main audio.
            - `--combination_volume`: Volume of the combined audio.
    """,
    "--help_audioldm2": """
    Parameters for `--audioldm2`:
        1. File Paths:
            - `--input_path` (required): Path to the input audio file.
            - `--output_path` (default: `./output.wav`): Path to save the output file.
            - `--export_format` (default: `wav`): Output file format.

        2. Audio Configuration:
            - `--sample_rate` (default: `44100`): Sampling rate (Hz).

        3. AudioLDM Model Configuration:
            - `--audioldm_model` (default: `audioldm2-music`): AudioLDM model to use.

        4. Model Prompt:
            - `--source_prompt` (default: ``): Source audio description.
            - `--target_prompt` (default: ``): Target audio description.

        5. Processing Algorithm:
            - `--steps` (default: `200`): Number of processing steps for audio synthesis.
            - `--cfg_scale_src` (default: `3.5`): Guidance scale for source audio.
            - `--cfg_scale_tar` (default: `12`): Guidance scale for target audio.
            - `--t_start` (default: `45`): Editing intensity.

        6. Computation Optimization:
            - `--save_compute` (default: `False`): Enable computation optimization.
    """,
    "--help_convert": """
    Parameters for `--convert`:
        1. Voice Processing:
            - `--pitch` (default: `0`): Pitch adjustment.
            - `--filter_radius` (default: `3`): F0 curve smoothness.
            - `--index_rate` (default: `0.5`): Voice index usage rate.
            - `--volume_envelope` (default: `1`): Volume amplitude adjustment factor.
            - `--protect` (default: `0.33`): Consonant protection level.

        2. Frame Hop Configuration:
            - `--hop_length` (default: `64`): Frame hop length for audio processing.

        3. F0 Configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, ..., `swipe`).
            - `--f0_autotune` (default: `False`): Enable/disable F0 autotuning.
            - `--f0_autotune_strength` (default: `1`): F0 autotune strength.
            - `--f0_file` (default: ``): Path to pre-existing F0 file.
            - `--f0_onnx` (default: `False`): Use ONNX version of F0.
            - `--proposal_pitch` (default: `False`): Suggest pitch instead of manual adjustment.

        4. Embedding Model:
            - `--embedder_model` (default: `contentvec_base`): Embedding model to use.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`).

        5. File Paths:
            - `--input_path` (required): Path to input audio file.
            - `--output_path` (default: `./audios/output.wav`): Path to save output file.
            - `--export_format` (default: `wav`): Output file format.
            - `--pth_path` (required): Path to `.pth` model file.
            - `--index_path` (default: `None`): Path to index file (if any).

        6. Audio Cleaning:
            - `--clean_audio` (default: `False`): Enable/disable audio cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning intensity.

        7. Resampling & Audio Splitting:
            - `--resample_sr` (default: `0`): New sampling rate (0 to keep original).
            - `--split_audio` (default: `False`): Enable/disable audio splitting before processing.

        8. Checkpointing & Optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.

        9. Formant Shifting:
            - `--formant_shifting` (default: `False`): Enable/disable formant shifting.
            - `--formant_qfrency` (default: `0.8`): Formant frequency shift factor.
            - `--formant_timbre` (default: `0.8`): Formant timbre adjustment factor.
    """,
    "--help_create_dataset": """
    Parameters for `--create_dataset`:
        1. Dataset Paths & Configuration:
            - `--input_audio` (required): Path or YouTube links to audio (use `,` for multiple links).
            - `--output_dataset` (default: `./dataset`): Output dataset directory.
            - `--sample_rate` (default: `44100`): Audio sampling rate.

        2. Data Cleaning:
            - `--clean_dataset` (default: `False`): Enable/disable dataset cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning intensity.

        3. Vocal Separation & Effects:
            - `--separator_reverb` (default: `False`): Enable/disable vocal reverb separation.
            - `--kim_vocal_version` (default: `2`): Kim Vocal model version (`1`, `2`).

        4. Audio Segmentation:
            - `--overlap` (default: `0.25`): Overlap between segments during separation.
            - `--segments_size` (default: `256`): Size of each audio segment.

        5. MDX (Music Demixing) Configuration:
            - `--mdx_hop_length` (default: `1024`): MDX hop length for processing.
            - `--mdx_batch_size` (default: `1`): Batch size for MDX processing.
            - `--denoise_mdx` (default: `False`): Enable/disable MDX denoising.

        6. Audio Skipping:
            - `--skip` (default: `False`): Enable/disable skipping parts of audio.
            - `--skip_start_audios` (default: `0`): Seconds to skip at the start of audio.
            - `--skip_end_audios` (default: `0`): Seconds to skip at the end of audio.
    """,
    "--help_create_index": """
    Parameters for `--create_index`:
        1. Model Information:
            - `--model_name` (required): Name of the model.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).
            - `--index_algorithm` (default: `Auto`): Index algorithm (`Auto`, `Faiss`, `KMeans`).
    """,
    "--help_extract": """
    Parameters for `--extract`:
        1. Model Information:
            - `--model_name` (required): Name of the model.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).

        2. F0 Configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, ..., `swipe`).
            - `--pitch_guidance` (default: `True`): Enable/disable pitch guidance.
            - `--f0_autotune` (default: `False`): Enable/disable F0 autotuning.
            - `--f0_autotune_strength` (default: `1`): F0 autotune strength.

        3. Processing Configuration:
            - `--hop_length` (default: `128`): Hop length for processing.
            - `--cpu_cores` (default: `2`): Number of CPU cores to use.
            - `--gpu` (default: `-`): Specify GPU to use (e.g., `0` for first GPU, `-` to disable GPU).
            - `--sample_rate` (required): Audio sampling rate.

        4. Embedding Configuration:
            - `--embedder_model` (default: `contentvec_base`): Embedding model name.
            - `--f0_onnx` (default: `False`): Use ONNx version of F0.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`).
    """,
    "--help_preprocess": """
    Parameters for `--preprocess`:
        1. Model Information:
            - `--model_name` (required): Name of the model.

        2. Data Configuration:
            - `--dataset_path` (default: `./dataset`): Path to dataset directory.
            - `--sample_rate` (required): Audio sampling rate.

        3. Processing Configuration:
            - `--cpu_cores` (default: `2`): Number of CPU cores to use.
            - `--cut_preprocess` (default: `True`): Enable/disable dataset file cutting.
            - `--process_effects` (default: `False`): Enable/disable preprocessing effects.
            - `--clean_dataset` (default: `False`): Enable/disable dataset cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning intensity.
    """,
    "--help_separator_music": """
    Parameters for `--separator_music`:
        1. Data Paths:
            - `--input_path` (required): Path to input audio file.
            - `--output_path` (default: `./audios`): Output directory for processed files.
            - `--format` (default: `wav`): Output file format (`wav`, `mp3`, etc.).

        2. Audio Processing Configuration:
            - `--shifts` (default: `2`): Number of predictions.
            - `--segments_size` (default: `256`): Audio segment size.
            - `--overlap` (default: `0.25`): Overlap between segments.
            - `--mdx_hop_length` (default: `1024`): MDX hop length for processing.
            - `--mdx_batch_size` (default: `1`): Batch size for processing.

        3. Cleaning Processing:
            - `--clean_audio` (default: `False`): Enable/disable audio cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning filter strength.

        4. Model Configuration:
            - `--model_name` (default: `HT-Normal`): Music separation model (`Main_340`, ..., `HT_6S`).
            - `--kara_model` (default: `Version-1`): Backing track separation model version (`Version-1`, `Version-2`).

        5. Effects & Post-Processing:
            - `--backing` (default: `False`): Enable/disable backing track separation.
            - `--mdx_denoise` (default: `False`): Enable/disable MDX denoising.
            - `--reverb` (default: `False`): Enable/disable reverb separation.
            - `--backing_reverb` (default: `False`): Enable/disable reverb separation for backing vocals.

        6. Sampling Rate:
            - `--sample_rate` (default: `44100`): Output audio sampling rate.
    """,
    "--help_train": """
    Parameters for `--train`:
        1. Model Configuration:
            - `--model_name` (required): Name of the model.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).
            - `--model_author` (optional): Model author name.

        2. Save Configuration:
            - `--save_every_epoch` (required): Number of epochs between saves.
            - `--save_only_latest` (default: `True`): Save only the latest checkpoint.
            - `--save_every_weights` (default: `True`): Save all model weights.

        3. Training Configuration:
            - `--total_epoch` (default: `300`): Total number of training epochs.
            - `--batch_size` (default: `8`): Batch size for training.
            - `--sample_rate` (required): Audio sampling rate.

        4. Device Configuration:
            - `--gpu` (default: `0`): Specify GPU to use (e.g., `0` for first GPU, `-` to disable).
            - `--cache_data_in_gpu` (default: `False`): Cache data in GPU for faster training.

        5. Advanced Training Configuration:
            - `--pitch_guidance` (default: `True`): Enable/disable pitch guidance.
            - `--g_pretrained_path` (default: ``): Path to pre-trained G weights.
            - `--d_pretrained_path` (default: ``): Path to pre-trained D weights.
            - `--vocoder` (default: `Default`): Vocoder to use (`Default`, `MRF-HiFi-GAN`, `RefineGAN`).

        6. Overtraining Detection:
            - `--overtraining_detector` (default: `False`): Enable/disable overtraining detection.
            - `--overtraining_threshold` (default: `50`): Threshold for overtraining detection.

        7. Data Processing:
            - `--cleanup` (default: `False`): Clean old training files to start fresh.

        8. Optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.
            - `--deterministic` (default: `False`): Use deterministic algorithms for reproducibility.
            - `--benchmark` (default: `False`): Test and select optimal algorithms for hardware.
            - `--optimizer` (default: `AdamW`): Optimizer to use (`AdamW`, `RAdam`).
    """,
    "--help": """
    Usage:
        1. `--help_audio_effects`: Help for adding audio effects.
        2. `--help_audioldm2`: Help for music editing.
        3. `--help_convert`: Help for audio conversion.
        4. `--help_create_dataset`: Help for creating a training dataset.
        5. `--help_create_index`: Help for creating an index.
        6. `--help_extract`: Help for extracting training data.
        7. `--help_preprocess`: Help for preprocessing data.
        8. `--help_separator_music`: Help for music separation.
        9. `--help_train`: Help for model training.
    """
}

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audio processing CLI tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=[cmd.name for cmd in COMMANDS] + list(HELP_COMMANDS.keys()),
        help="Command to execute or help to display"
    )
    return parser

def import_main_module(module_path: str) -> Callable:
    """Dynamically import the main function from the specified module."""
    module = __import__(module_path, fromlist=["main"])
    return module.main

def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Handle help commands
    if args.command in HELP_COMMANDS:
        print(HELP_COMMANDS[args.command])
        sys.exit(0)

    # Find and execute the command
    for cmd in COMMANDS:
        if args.command == cmd.name:
            try:
                main_func = import_main_module(cmd.module)
                # Set multiprocessing start method for specific commands
                if args.command in ["--train", "--preprocess", "--extract"]:
                    import torch.multiprocessing as mp
                    mp.set_start_method("spawn", force=args.command in ["--preprocess", "--extract"])
                main_func()
                return
            except ImportError as e:
                print(f"Error: Failed to import module for {cmd.name}: {e}")
                sys.exit(1)

    print("Invalid command! Use --help for more information.")
    sys.exit(1)

if __name__ == "__main__":
    main()
