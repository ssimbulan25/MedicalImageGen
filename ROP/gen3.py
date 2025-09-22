
from torch.utils.data import Dataset
import torch.nn as nn

from datasets import load_dataset
from dataclasses import dataclass
from torchvision import transforms
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from transformers import T5Tokenizer, T5EncoderModel

from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import math
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from pathlib import Path
import os
from huggingface_hub import create_repo, upload_folder
import glob
from huggingface_hub import login


login(token = "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")

device = torch.device("cuda:1")
torch.set_default_device(device)


@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    in_channels = 1
    out_channels = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'SD3ControlNet'  # the model namy locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = True  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        
    ]
)
preprocess_mask = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])

def transform_img(examples):
    images = [preprocess(image.convert("L")) for image in examples["image"]]
    return {"images": images}

def transform_mask(examples): 
    images = [preprocess_mask(image.convert("L")) for mask in examples["mask"]]
    return {"images": images}

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=transform_img, transform_mask=transform_mask):
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask  = Image.open(self.mask_paths[idx]).convert("L")  # grayscale mask

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return {"images": image, "masks": mask}
    
image_dir = "/media/yuganlab/blackstone/HZL/Data/images"
mask_dir  = "/media/yuganlab/blackstone/HZL/Data/labels"

train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=preprocess,  # your image transform
    transform_mask=preprocess_mask  # mask transform
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
controlnet.to(device)

# sample_mask = train_dataset[0]['masks'].unsqueeze(0)
# sample_image = train_dataset[0]['images'].unsqueeze(0)

scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config.learning_rate)




lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)



# def make_grid(images, rows, cols):
#     w, h = images[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     for i, image in enumerate(images):
#         grid.paste(image, box=(i%cols*w, i//cols*h))
#     return grid

def evaluate(config, epoch, pipe):
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    images = pipe(prompt="malignant tumor", image=test_mask_tensor.to(device), generator=generator.to(device)).images
    images[0].save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, controlnet, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    
    logging_dir = os.path.join(config.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(config.output_dir).name, exist_ok=True
            ).repo_id
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    
    generator = torch.Generator(device=device).manual_seed(24)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    text_encoder = T5EncoderModel.from_pretrained("t5-base").to(device)
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(device)
            
            if clean_images.shape[1] == 1:
                clean_images = clean_images.repeat(1, 16, 1, 1)
                
            seg_masks = batch['masks'].to(dtype=torch.float32, device=device) # [B, 1, H, W]
            seg_masks = seg_masks.repeat(1, 16, 1, 1)
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device, generator=generator)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(controlnet):
                # Predict the noise residual
                text_inputs = tokenizer(
                    ["a photo of a tumor"],  # or any relevant prompt
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(**text_inputs).last_hidden_state
                    encoder_hidden_states.to(device)
                    
                noise_pred = controlnet(
                    noisy_images,
                    timesteps,
                    seg_masks, 
                    encoder_hidden_states=encoder_hidden_states
                ).images[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                controlnet=controlnet
            ).to(device)

            pipe.text_encoder.to(device=device)
            pipe.controlnet.to(device=device)
            pipe.to(device=device)
            # Run inference after moving model and data to GPU
            
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipe)
                pipe.save_pretrained(config.output_dir)

args = (config, controlnet, scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])