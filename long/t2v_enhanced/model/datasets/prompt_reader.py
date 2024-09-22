from pathlib import Path
from typing import Dict, List, Optional
import csv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import random 
from t2v_enhanced.model.datasets.video_dataset import Annotations
import json
import pandas as pd
import torchvision.transforms as transforms
from decord import VideoReader
from datasets import load_dataset

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.model_id = datasets["reconstruction_dataset"].model_id

    def __getitem__(self, idx):
        sample = {ds: self.datasets[ds].__getitem__(
            idx) for ds in self.datasets}
        return sample

    def __len__(self):
        return min(len(self.datasets[d]) for d in self.datasets)


class CustomPromptsDataset(torch.utils.data.Dataset):

    def __init__(self, prompt_cfg: Dict[str, str], sample_n_frames=16):
        super().__init__()
        self.sample_n_frames = sample_n_frames

        if prompt_cfg["type"] == "prompt":
            self.prompts = [prompt_cfg["content"]]
            transformed_prompts = []
            for prompt in self.prompts:
                transformed_prompts.append(
                    Annotations.clean_prompt(prompt))
            self.prompts = transformed_prompts
        elif prompt_cfg["type"] == "file":
            file = Path(prompt_cfg["content"])
            if file.suffix == ".npy":
                self.prompts = np.load(file.as_posix())
            elif file.suffix == ".txt":
                with open(prompt_cfg["content"]) as f:
                    lines = [line.rstrip() for line in f]
                self.prompts = lines
            elif file.suffix == ".json":
                with open(prompt_cfg["content"],"r") as file:
                    metadata = json.load(file)
                if "videos_root" in prompt_cfg:
                    videos_root = Path(prompt_cfg["videos_root"])
                    video_path = [str(videos_root / sample["page_dir"] /
                                  f"{sample['videoid']}.mp4") for sample in metadata]
                else:
                    video_path = [str(sample["page_dir"] /
                                  f"{sample['videoid']}.mp4") for sample in metadata]
                self.prompts = [sample["prompt"] for sample in metadata]
                self.video_path = video_path

            transformed_prompts = []
            for prompt in self.prompts:
                transformed_prompts.append(
                    Annotations.clean_prompt(prompt))
            self.prompts = transformed_prompts
        elif prompt_cfg["type"] == "csv":
            csv_path = Path(prompt_cfg["content"])
            # self.dataset = pd.read_csv(csv_path)
            # with open(csv_path, "r") as csv_file:
            #     self.dataset = list(csv.DictReader(csv_file))
            self.dataset = load_dataset('csv', data_files=csv_path.as_posix()).filter(lambda x: 30 <= x["FlowScore"] <= 60)['train']

        sample_size = (256,256)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        

    def __len__(self):
        if hasattr(self, "dataset"):
            return len(self.dataset)
        return len(self.prompts)
    
    def get_batch(self, index):
        video_dict = self.dataset[index]
        video_path, prompt = video_dict['video_path'], video_dict['caption']
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames-1) * 4 + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).numpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        pixel_values = self.pixel_transforms(pixel_values)
        del video_reader
        return {"prompt": prompt, "pixel_values": pixel_values}




    def __getitem__(self, index):
        if hasattr(self, "dataset"):
            while True:
                try:
                    # print(self.get_batch(index))
                    return self.get_batch(index)
                except Exception as e:
                    index = random.randint(0, len(self.dataset)-1)
                    print(e, f"Not found! Trying another index: {index}")

        output = {"prompt": self.prompts[index]}
        if hasattr(self,"video_path"):
            output["video"] = self.video_path[index]
        return output


class PromptReader(pl.LightningDataModule):
    def __init__(self, prompt_cfg: Dict[str, str], sample_n_frames: int = 16):
        super().__init__()
        self.predict_dataset = CustomPromptsDataset(prompt_cfg, sample_n_frames)
        # self.train_dataset = CustomPromptsDataset(prompt_cfg)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=1, pin_memory=False, shuffle=False, drop_last=False)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=2, pin_memory=False, shuffle=True, drop_last=False, num_workers=1)