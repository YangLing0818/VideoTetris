
## ___***VideoTetris: Towards Compositional Text-To-Video Generation***___
<div align="left">
 <a href='https://arxiv.org/abs/2406.04277'><img src='https://img.shields.io/badge/arXiv-2406.04277-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://videotetris.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

This repo contains the official implementation of our [VideoTetris](https://arxiv.org/abs/2406.04277).

> [**VideoTetris: Towards Compositional Text-To-Video Generation**](https://arxiv.org/abs/2406.04277)   
> [Ye Tian](https://tyfeld.github.io/),
> [Ling Yang*](https://yangling0818.github.io), 
> [Haotian Yang](https://scholar.google.com/citations?user=LH71RGkAAAAJ&hl=en),
> [Yuan Gao](https://videotetris.github.io/),
> [Yufan Deng](https://videotetris.github.io/),
> [Jingmin Chen](https://videotetris.github.io/),
> [Xintao Wang](https://xinntao.github.io),
> [Zhaochen Yu](https://videotetris.github.io/),
> [Pengfei Wan](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en),
> [Di Zhang](https://openreview.net/profile?id=~Di_ZHANG3),
> [Bin Cui](https://cuibinpku.github.io/cuibin_cn.html)   
> (* Equal Contribution and Corresponding Author)
> <br>Peking University, Kuaishou Technology<br>

## News Update
- [x] Paper [VideoTetris](https://arxiv.org/abs/2406.04277) released 
- [x] Release our inference code of VideoTetris for compositioanl text-to-video generation
- [x] Release our finetuned StreamingT2V for long video generation based on our filtered dataset
- [x] Release the training/inference code of long compositional video generation 
- [x] Release the checkpoint of our long compositional video generation 

## Introduction
VideoTetris is a novel framework that enables **compositional T2V generation**. Specifically, we propose **spatio-temporal compositional diffusion** to precisely follow complex textual semantics by manipulating and composing the attention maps of denoising networks spatially and temporally. Moreover, we propose an enhanced video data preprocessing to enhance the training data regarding motion dynamics and prompt understanding, equipped with a new reference frame attention mechanism to improve the consistency of auto-regressive video generation.  Our demonstrations include successful examples of **videos spanning from 10s, 30s to 2 minutes**, and can be extended for even longer durations.
<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="assets/first.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">
</td>
  </tr>
    <tr>
    <td width=100% style="border: none"><img src="assets/secondd.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">
</td>
  </tr>
</table>




## Training and Inference

### Composition Text-to-Video Generation
We provide the inference code of our VideoTetris for compositional video generation based on VideoCrafter2. You can download the pretrained model from [Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) and put it in `checkpoints/base_512_v2/model.ckpt`. Then run the following command:
#### 1. Install Environment via Anaconda (Recommended)
```bash
cd short
conda create -n videocrafter python=3.8.5
conda activate videocrafter
pip install -r requirements.txt
```

#### 2. Region Planning
You can then plan the regions for different sub-objects in a json file like `prompts/demo_videotetris.json`. The regions are defined by the top-left and bottom-right coordinates of the bounding box. You can refer to the `prompts/demo_videotetris.json` for an example. And the final planning json should be like:
```json
{
  {
    "basic_prompt": "A cat on the left and a dog on the right are napping in the sun.",
    "sub_objects":[
        "A cute orange cat.",
        "A cute dog."
    ],
    "layout_boxes":[
        [0, 0, 0.5, 1],
        [0.5, 0, 1, 1]
    ]
  },
}
```
In this case, we first define the basic prompt, and then specify the sub-objects and their corresponding regions, resulting in a video with a left cat and a right dog.

#### 3. Inference of VideoTetris
```bash
sh scripts/run_text2video_from_layout.sh
```
You can specify the input json file `run_text2video_from_layout.sh` script.


### Long Video Generation with Progressive Compositional Prompts

#### 1. Install Environment via Anaconda (Recommended)
We build our long video generation code based on [StreamingT2V](https://github.com/Picsart-AI-Research/StreamingT2V). The corresponding environment of StreamingT2V is:
```bash
cd long
conda create -n st2v python=3.10
conda activate st2v
pip install -r requirements.txt
```
#### 2. Download the Checkpoint

We put our VideoTetris-long model finetuned on our filtered dataset on [Hugging Face](https://huggingface.co/tyfeld/VideoTetris-long). You can download the weights and put it in the directory through:
```bash
wget https://huggingface.co/tyfeld/VideoTetris-long/resolve/main/model-step=6000-v1.ckpt
```

#### 3. Region Planning

You can then plan the regions for different sub-objects in a json file like prompts/prompt.json. You should specify the video chunk index, prompt, sub-objects and layout boxes for each video chunk. 

> Video Chunk Meaning: As the long video is autoregressively generated by 8 frames for each chunk, a video with 80 frames will be autoregressively generated with (80-8)/8 = 9 rounds, as defined in the original StreamingT2V repo. And every chunk means the expanding 8 frames generated in one round.

The regions are defined by the top-left and bottom-right coordinates of the bounding box. You can refer to the prompts/prompt.json for an example. And the final planning json should be like:
```json
[
    {
        "video_chunk_index": 0, 
        "prompt": "A cute brown squirrel in Antarctica, on a pile of hazelnuts cinematic.",
        "sub_objects": [
            "A cute brown squirrel in Antarctica, on a pile of hazelnuts cinematic."
        ],
        "layout_boxes":[
            [0, 0, 1, 1]
        ]
    },
    {
        "video_chunk_index": 4,
        "prompt": "A cute brown squirrel and a cute white squirrel in Antarctica, on a pile of hazelnuts cinematic",
        "sub_objects": [
            "A cute brown squirrel in Antarctica, on a pile of hazelnuts cinematic.",
            "A cute white squirrel in Antarctica, on a pile of hazelnuts cinematic."
        ],
        "layout_boxes":[
            [0.5, 0, 1, 1],
            [0, 0, 0.5, 1]
        ]
    }
]
```
#### 4. Inference of VideoTetris-long
```bash
cd t2v_enhanced
python inference_videotetris.py --num_frames 80
```

## Example Results
We only provide some example results here, more detailed results can be found in the [project page](https://videotetris.github.io/).
<table class="center">
    <tr>
    <td width=25% style="border: none"><img src="assets/cat_and_dog.gif" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/farmer_and_blacksmith.gif" style="width:100%"></td>
  <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A cute brown dog on the left and a sleepy cat on the right are napping in the sun. <br> @16 Frames</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A cheerful farmer and a hardworking blacksmith are building a barn. <br> @16 Frames</td>
  </tr>
</table>

<table class="center">
    <tr>
    <td width=35% style="border: none"><img src="assets/1234.gif" style="width:130%"></td>
    <td width=35% style="border: none"><img src="assets/brown2white.gif" style="width:130%"></td>
  <tr>
    <td width="35%" style="border: none; text-align: center; word-wrap: break-word">One cute brown squirrel, on a pile of hazelnuts, cinematic. <br> ------>  transitions to <br>
Two cute brown squirrels, on a pile of hazelnuts, cinematic. <br> ------>  transitions to <br>
Three cute brown squirrels, on a pile of hazelnuts, cinematic. <br> ------>  transitions to <br>
Four cute brown squirrels, on a pile of hazelnuts, cinematic. <br> 
 @80 Frames</td>
    <td width="35%" style="border: none; text-align: center; word-wrap: break-word">A cute brown squirrel, on a pile of hazelnuts, cinematic. <br> ------>  transitions to <br>
A cute brown squirrel and a cute white squirrel, on a pile of hazelnuts, cinematic.  <br>
 @240 Frames</td>
  </tr>
</table>




## Citation
```
@article{tian2024videotetris,
  title={VideoTetris: Towards Compositional Text-to-Video Generation},
  author={Tian, Ye and Yang, Ling and Yang, Haotian and Gao, Yuan and Deng, Yufan and Chen, Jingmin and Wang, Xintao and Yu, Zhaochen and Tao, Xin and Wan, Pengfei and Zhang, Di and Cui, Bin},
  journal={arXiv preprint arXiv:2406.04277},
  year={2024}
}
```

