import os
import sys

sys.path.append(os.getcwd())

try:
    argv = sys.argv[1]
except IndexError:
    argv = None

argv_is_allows = ["--audio_effects", "--audioldm2", "--convert", "--create_dataset", "--create_index", "--extract", "--preprocess", "--separator_music", "--train", "--help_audio_effects", "--help_audioldm2", "--help_convert", "--help_create_dataset", "--help_create_index", "--help_extract", "--help_preprocess", "--help_separator_music", "--help_train", "--help"]

if argv not in argv_is_allows:
    print("Invalid syntax! Use --help for more information")
    quit()

if argv_is_allows[0] in argv: from main.inference.audio_effects import main
elif argv_is_allows[1] in argv: from main.inference.audioldm2 import main
elif argv_is_allows[2] in argv: from main.inference.convert import main
elif argv_is_allows[3] in argv: from main.inference.create_dataset import main
elif argv_is_allows[4] in argv: from main.inference.create_index import main
elif argv_is_allows[5] in argv: from main.inference.extract import main
elif argv_is_allows[6] in argv: from main.inference.preprocess import main
elif argv_is_allows[7] in argv: from main.inference.separator_music import main
elif argv_is_allows[8] in argv: from main.inference.train import main
elif argv_is_allows[9] in argv:
    print("""Parameters for `--audio_effects`:
        1. File paths:
            - `--input_path` (required): Path to the input audio file.
            - `--output_path` (default: `./audios/apply_effects.wav`): Path to save the output file.
            - `--export_format` (default: `wav`): Output file format (`wav`, `mp3`, ...).

        2. Resampling:
            - `--resample` (default: `False`): Whether to resample or not.
            - `--resample_sr` (default: `0`): New sampling frequency (Hz).

        3. Chorus effect:
            - `--chorus`: Enable/disable chorus.
            - `--chorus_depth`, `--chorus_rate`, `--chorus_mix`, `--chorus_delay`, `--chorus_feedback`: Parameters to adjust chorus.

        4. Distortion effect:
            - `--distortion`: Enable/disable distortion.
            - `--drive_db`: Degree of audio distortion.

        5. Reverb effect:
            - `--reverb`: Enable/disable reverb.
            - `--reverb_room_size`, `--reverb_damping`, `--reverb_wet_level`, `--reverb_dry_level`, `--reverb_width`, `--reverb_freeze_mode`: Adjust reverb.

        6. Pitch shift effect:
            - `--pitchshift`: Enable/disable pitch shift.
            - `--pitch_shift`: Pitch shift value.

        7. Delay effect:
            - `--delay`: Enable/disable delay.
            - `--delay_seconds`, `--delay_feedback`, `--delay_mix`: Adjust delay time, feedback, and mix.

        8. Compressor:
            - `--compressor`: Enable/disable compressor.
            - `--compressor_threshold`, `--compressor_ratio`, `--compressor_attack_ms`, `--compressor_release_ms`: Compression parameters.

        9. Limiter:
            - `--limiter`: Enable/disable audio level limiter.
            - `--limiter_threshold`, `--limiter_release`: Limiter threshold and release time.

        10. Gain (Amplification):
            - `--gain`: Enable/disable gain.
            - `--gain_db`: Gain level (dB).

        11. Bitcrush:
            - `--bitcrush`: Enable/disable bit resolution reduction effect.
            - `--bitcrush_bit_depth`: Bit depth for bitcrush.

        12. Clipping:
            - `--clipping`: Enable/disable audio clipping.
            - `--clipping_threshold`: Clipping threshold.

        13. Phaser:
            - `--phaser`: Enable/disable phaser effect.
            - `--phaser_rate_hz`, `--phaser_depth`, `--phaser_centre_frequency_hz`, `--phaser_feedback`, `--phaser_mix`: Adjust phaser effect.

        14. Boost bass & treble:
            - `--treble_bass_boost`: Enable/disable bass and treble boost.
            - `--bass_boost_db`, `--bass_boost_frequency`, `--treble_boost_db`, `--treble_boost_frequency`: Bass and treble boost parameters.

        15. Fade in & fade out:
            - `--fade_in_out`: Enable/disable fade effect.
            - `--fade_in_duration`, `--fade_out_duration`: Fade in/out duration.

        16. Audio combination:
            - `--audio_combination`: Enable/disable combining multiple audio files.
            - `--audio_combination_input`: Path to additional audio files.
    """)
    quit()
elif argv_is_allows[10] in argv:
    print("""Parameters for `--audioldm2`:
        1. File paths:
            - `--input_path` (required): Path to the input audio file.
            - `--output_path` (default: `./output.wav`): Path to save the output file.
            - `--export_format` (default: `wav`): Output file format.

        2. Audio configuration:
            - `--sample_rate` (default: `44100`): Sampling frequency (Hz).

        3. AudioLDM model configuration:
            - `--audioldm_model` (default: `audioldm2-music`): Select AudioLDM model for processing.

        4. Model guidance prompt:
            - `--source_prompt` (default: ``): Description of source audio.
            - `--target_prompt` (default: ``): Description of target audio.

        5. Processing algorithm configuration:
            - `--steps` (default: `200`): Number of steps in audio synthesis process.
            - `--cfg_scale_src` (default: `3.5`): Guidance scale for source audio.
            - `--cfg_scale_tar` (default: `12`): Guidance scale for target audio.
            - `--t_start` (default: `45`): Editing level.

        6. Computation optimization:
            - `--save_compute` (default: `False`): Whether to enable compute optimization mode.
    """)
    quit()
elif argv_is_allows[11] in argv:
    print("""Parameters for `--convert`:
        1. Voice processing configuration:
            - `--pitch` (default: `0`): Adjust pitch.
            - `--filter_radius` (default: `3`): F0 curve smoothness.
            - `--index_rate` (default: `0.5`): Voice index usage rate.
            - `--volume_envelope` (default: `1`): Volume amplitude adjustment factor.
            - `--protect` (default: `0.33`): Consonant protection.

        2. Frame hop configuration:
            - `--hop_length` (default: `64`): Hop length during audio processing.

        3. F0 configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_autotune` (default: `False`): Whether to auto-tune F0.
            - `--f0_autotune_strength` (default: `1`): Strength of F0 auto-tuning.
            - `--f0_file` (default: ``): Path to existing F0 file.
            - `--f0_onnx` (default: `False`): Whether to use ONNX version of F0.

        4. Embedding model:
            - `--embedder_model` (default: `contentvec_base`): Embedding model used.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`).

        5. File paths:
            - `--input_path` (required): Path to input audio file.
            - `--output_path` (default: `./audios/output.wav`): Path to save output file.
            - `--export_format` (default: `wav`): Output file format.
            - `--pth_path` (required): Path to `.pth` model file.
            - `--index_path` (default: `None`): Path to index file (if any).

        6. Audio cleaning:
            - `--clean_audio` (default: `False`): Whether to apply audio cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning strength.

        7. Resampling & audio splitting:
            - `--resample_sr` (default: `0`): New sampling frequency (0 means keep original).
            - `--split_audio` (default: `False`): Whether to split audio before processing.

        8. Testing & optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.

        9. Formant shifting:
            - `--formant_shifting` (default: `False`): Whether to enable formant shifting effect.
            - `--formant_qfrency` (default: `0.8`): Formant shift frequency factor.
            - `--formant_timbre` (default: `0.8`): Voice timbre change factor.
    """)
    quit()
elif argv_is_allows[12] in argv:
    print("""Parameters for `--create_dataset`:
        1. Dataset paths & configuration:
            - `--input_audio` (required): Path to audio link (YouTube link, can use `,` for multiple links).
            - `--output_dataset` (default: `./dataset`): Output data directory.
            - `--sample_rate` (default: `44100`): Audio sampling frequency.

        2. Data cleaning:
            - `--clean_dataset` (default: `False`): Whether to apply data cleaning.
            - `--clean_strength` (default: `0.7`): Data cleaning strength.

        3. Voice separation & effects:
            - `--separator_reverb` (default: `False`): Whether to separate voice reverb.
            - `--kim_vocal_version` (default: `2`): Kim Vocal model version for separation (`1`, `2`).

        4. Audio segmentation configuration:
            - `--overlap` (default: `0.25`): Overlap level between segments during separation.
            - `--segments_size` (default: `256`): Size of each segment.

        5. MDX (Music Demixing) configuration:
            - `--mdx_hop_length` (default: `1024`): MDX hop length during processing.
            - `--mdx_batch_size` (default: `1`): Batch size during MDX processing.
            - `--denoise_mdx` (default: `False`): Whether to apply denoising during MDX separation.

        6. Skip audio sections:
            - `--skip` (default: `False`): Whether to skip any audio seconds.
            - `--skip_start_audios` (default: `0`): Time (seconds) to skip at the start of audio.
            - `--skip_end_audios` (default: `0`): Time (seconds) to skip at the end of audio.
    """)
    quit()
elif argv_is_allows[13] in argv:
    print("""Parameters for `--create_index`:
        1. Model information:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): Version (`v1`, `v2`).
            - `--index_algorithm` (default: `Auto`): Index algorithm used (`Auto`, `Faiss`, `KMeans`).
    """)
    quit()
elif argv_is_allows[14] in argv:
    print("""Parameters for `--extract`:
        1. Model information:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).

        2. F0 configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--pitch_guidance` (default: `True`): Whether to use pitch guidance.

        3. Processing configuration:
            - `--hop_length` (default: `128`): Hop length during processing.
            - `--cpu_cores` (default: `2`): Number of CPU threads used.
            - `--gpu` (default: `-`): Specify GPU to use (e.g., `0` for first GPU, `-` to disable GPU).
            - `--sample_rate` (required): Input audio sampling frequency.

        4. Embedding configuration:
            - `--embedder_model` (default: `contentvec_base`): Embedding model name.
            - `--f0_onnx` (default: `False`): Whether to use ONNX version of F0.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`).
    """)
    quit()
elif argv_is_allows[15] in argv:
    print("""Parameters for `--preprocess`:
        1. Model information:
            - `--model_name` (required): Model name.

        2. Data configuration:
            - `--dataset_path` (default: `./dataset`): Path to directory containing data files.
            - `--sample_rate` (required): Audio data sampling frequency.

        3. Processing configuration:
            - `--cpu_cores` (default: `2`): Number of CPU threads used.
            - `--cut_preprocess` (default: `True`): Whether to cut data files.
            - `--process_effects` (default: `False`): Whether to apply preprocessing.
            - `--clean_dataset` (default: `False`): Whether to clean data files.
            - `--clean_strength` (default: `0.7`): Data cleaning strength.
    """)
    quit()
elif argv_is_allows[16] in argv:
    print("""Parameters for `--separator_music`:
        1. Data paths:
            - `--input_path` (required): Path to input audio file.
            - `--output_path` (default: `./audios`): Directory to save output files.
            - `--format` (default: `wav`): Output file format (`wav`, `mp3`, ...).

        2. Audio processing configuration:
            - `--shifts` (default: `2`): Number of predictions.
            - `--segments_size` (default: `256`): Audio segment size.
            - `--overlap` (default: `0.25`): Overlap level between segments.
            - `--mdx_hop_length` (default: `1024`): MDX hop length during processing.
            - `--mdx_batch_size` (default: `1`): Batch size.

        3. Cleaning processing:
            - `--clean_audio` (default: `False`): Whether to clean audio.
            - `--clean_strength` (default: `0.7`): Cleaning filter strength.

        4. Model configuration:
            - `--model_name` (default: `HT-Normal`): Music separation model (`Main_340`, `Main_390`, `Main_406`, `Main_427`, `Main_438`, `Inst_full_292`, `Inst_HQ_1`, `Inst_HQ_2`, `Inst_HQ_3`, `Inst_HQ_4`, `Inst_HQ_5`, `Kim_Vocal_1`, `Kim_Vocal_2`, `Kim_Inst`, `Inst_187_beta`, `Inst_82_beta`, `Inst_90_beta`, `Voc_FT`, `Crowd_HQ`, `Inst_1`, `Inst_2`, `Inst_3`, `MDXNET_1_9703`, `MDXNET_2_9682`, `MDXNET_3_9662`, `Inst_Main`, `MDXNET_Main`, `MDXNET_9482`, `HT-Normal`, `HT-Tuned`, `HD_MMI`, `HT_6S`).
            - `--kara_model` (default: `Version-1`): Backing track separation model version (`Version-1`, `Version-2`).

        5. Effects and post-processing:
            - `--backing` (default: `False`): Whether to separate backing vocals.
            - `--mdx_denoise` (default: `False`): Whether to use MDX denoising.
            - `--reverb` (default: `False`): Whether to separate reverb.
            - `--backing_reverb` (default: `False`): Whether to separate reverb for backing vocals.

        6. Sampling frequency:
            - `--sample_rate` (default: `44100`): Output audio sampling frequency.
    """)
    quit()
elif argv_is_allows[17] in argv:
    print("""Parameters for `--train`:
        1. Model configuration:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).
            - `--model_author` (optional): Model author.

        2. Save configuration:
            - `--save_every_epoch` (required): Number of epochs between saves.
            - `--save_only_latest` (default: `True`): Save only the latest checkpoint.
            - `--save_every_weights` (default: `True`): Save all model weights.

        3. Training configuration:
            - `--total_epoch` (default: `300`): Total number of training epochs.
            - `--batch_size` (default: `8`): Batch size during training.
            - `--sample_rate` (required): Audio sampling frequency.

        4. Device configuration:
            - `--gpu` (default: `0`): Specify GPU to use (agena: Specify GPU to use (number or `-` if not using GPU).
            - `--cache_data_in_gpu` (default: `False`): Cache data in GPU for faster processing.

        5. Advanced training configuration:
            - `--pitch_guidance` (default: `True`): Use pitch guidance.
            - `--g_pretrained_path` (default: ``): Path to pretrained G weights.
            - `--d_pretrained_path` (default: ``): Path to pretrained D weights.
            - `--vocoder` (default: `Default`): Vocoder used (`Default`, `MRF-HiFi-GAN`, `RefineGAN`).

        6. Overtraining detection:
            - `--overtraining_detector` (default: `False`): Enable/disable overtraining detection.
            - `--overtraining_threshold` (default: `50`): Threshold for detecting overtraining.

        7. Data processing:
            - `--cleanup` (default: `False`): Clean old training files to start training from scratch.

        8. Optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.
            - `--deterministic` (default: `False`): When enabled, uses deterministic algorithms to ensure consistent results for the same input data.
            - `--benchmark` (default: `False`): When enabled, tests and selects the optimal algorithm for the hardware and specific size.
    """)
    quit()
elif argv_is_allows[18] in argv:
    print("""Usage:
        1. `--help_audio_effects`: Help for adding audio effects.
        2. `--help_audioldm2`: Help for music editing.
        3. `--help_convert`: Help for audio conversion.
        4. `--help_create_dataset`: Help for creating training data.
        5. `--help_create_index`: Help for creating an index.
        6. `--help_extract`: Help for extracting training data.
        7. `--help_preprocess`: Help for preprocessing data.
        8. `--help_separator_music`: Help for music separation.
        9. `--help_train`: Help for model training.
    """)
    quit()

if __name__ == "__main__":
    if "--train" in argv:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
        
    try:
        main()
    except:
        pass
