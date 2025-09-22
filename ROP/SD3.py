
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from transformers import T5Tokenizer, T5EncoderModel
from torch.optim import AdamW
from pathlib import Path
import os
import sys
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import math
from accelerate import Accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ControlNet')))

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict



torch.cuda.set_device(1)

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    in_channels = 3
    out_channels = 3
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'SD3ControlNet'  # the model namy locally and on the HF Hub

config = TrainingConfig()
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        
    ]
)
preprocess_mask = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])

def transform_img(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["images"]]
    return {"images": images}

def transform_mask(examples): 
    masks = [preprocess_mask(mask.convert("L")) for mask in examples["masks"]]
    return {"masks": masks}

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=transform_img, transform_mask=transform_mask, prompt="a photo of an eye with visible vascular structure"):
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        self.prompt = prompt
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        prompt = self.prompt

        if self.transform_img:
            image = self.transform_img(image)  # Now works correctly
        if self.transform_mask:
            mask = self.transform_mask(mask)
        



        return {
            "images": image,
            "masks": mask,
            "txt": prompt  # use "txt" to match ControlNet text key
        }
    
image_dir = "/media/yuganlab/blackstone/HZL/SegDiff/fundus/data_folder/test"
mask_dir  = "/media/yuganlab/blackstone/HZL/SegDiff/fundus/mask_folder/all/test"

train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=preprocess,  # your image transform
    transform_mask=preprocess_mask  # mask transform
)

max_cpus = multiprocessing.cpu_count()
num_workers = min(max_cpus // 2, 8)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    drop_last=True, 
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True)


model = create_model('/media/yuganlab/blackstone/ControlNet/models/cldm_v15.yaml').cpu()
state_dict = torch.load('/media/yuganlab/blackstone/ControlNet/models/control_sd15_seg.pth')
model.load_state_dict(state_dict, strict=False)
model.learning_rate = config.learning_rate
model.sd_locked = True
model.only_mid_control = False


# Train!
logger = ImageLogger(batch_frequency=300)
trainer = pl.Trainer(accelerator="gpu", devices=[1], precision=32, callbacks=[logger])

trainer.fit(model, train_dataloader)
torch.save(model.state_dict(), "model_weights.pth")

