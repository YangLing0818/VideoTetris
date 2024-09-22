# General
import os
import sys
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import join as opj
import argparse
import datetime
from pathlib import Path
import torch
import gradio as gr
import tempfile
import yaml
from t2v_enhanced.model.video_ldm import VideoLDM
from typing import List, Optional
from model.callbacks import SaveConfigCallback
from PIL.Image import Image, fromarray
from einops import rearrange, repeat

sys.path.append("thirdparty")
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import imageio
import pathlib
import numpy as np

# Utilities
from inference_utils import *
from model_init import *
from model_func import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, default="prompts/prompt.json", help="The prompt json file to guide video generation.")
    parser.add_argument('--num_frames', type=int, default=24, help="The number of video frames to generate.")
    parser.add_argument('--negative_prompt', type=str, default="", help="The prompt to guide what to not include in video generation.")
    parser.add_argument('--negative_prompt_enhancer', type=str, default=None, help="The prompt to guide what to not include in video enhancement. "
                        "By default is the same as --negative_prompt")
    parser.add_argument('--num_steps', type=int, default=50, help="The number of denoising steps.")
    parser.add_argument('--image_guidance', type=float, default=9.0, help="The guidance scale.")

    parser.add_argument('--output_dir', type=str, default="results", help="Path where to save the generated videos.")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=33, help="Random seed")
    
    parser.add_argument('--offload_models', action='store_true', help="Load/Offload models to gpu/cpu before and after inference")
    args = parser.parse_args()


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_fol = Path(args.output_dir).absolute()
    device = args.device

    ckpt_file_streaming_t2v = Path("model-step=6000-v1.ckpt").absolute()
    cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}

    # --------------------------
    # ----- Initialization -----
    # --------------------------
    if args.offload_models:
        model = init_modelscope("cpu")
    else:
        model = init_modelscope(device)

    stream_cli, stream_model = init_streamingt2v_model(ckpt_file_streaming_t2v, result_fol, device)
    if args.offload_models:
        stream_model = st2v_to_device(stream_model, "cpu")
    inference_generator = torch.Generator(device=device)

    # ------------------
    # ----- Inputs -----
    # ------------------
    comp_prompt = load_prompts_from_json(args.prompt_file)
    first_prompt = comp_prompt[0]["prompt"]
    now = datetime.datetime.now()
    name = first_prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

    inference_generator = torch.Generator(device=device)
    inference_generator.manual_seed(args.seed)

    if args.offload_models:
        model = model.to(device)
    short_video = ms_short_gen(first_prompt, model, inference_generator)
    if args.offload_models:
        model = model.to("cpu")

    n_autoreg_gen = (args.num_frames-8)//8
    comp_prompt, change_point = fill_video_chunks(comp_prompt, n_autoreg_gen)
    stream_long_gen(comp_prompt, change_point, short_video, n_autoreg_gen, args.negative_prompt, args.seed, args.num_steps, args.image_guidance, name, stream_cli, stream_model)
