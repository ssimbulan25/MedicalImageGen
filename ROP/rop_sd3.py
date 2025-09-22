import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from torchvision import transforms
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from torch.utils.data import random_split
import multiprocessing
from transformers import CLIPTokenizer, CLIPTextModel

from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import math
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from pathlib import Path
from huggingface_hub import create_repo, upload_folder
import glob
from huggingface_hub import login

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login(token = "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")

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
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
    
image_dir = "/media/yuganlab/blackstone/HZL/Retinal_seg/Retina-Blood-Vessel-Segmentation-in-PyTorch/data/train/image"
mask_dir  = "/media/yuganlab/blackstone/HZL/Retinal_seg/Retina-Blood-Vessel-Segmentation-in-PyTorch/data/train/mask"

dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=preprocess,  # your image transform
    transform_mask=preprocess_mask  # mask transform
)


val_frac = 0.1
n_val = int(len(dataset) * val_frac)
n_train = len(dataset) - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

max_cpus = multiprocessing.cpu_count()
num_workers = min(max_cpus // 2, 8)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=config.train_batch_size, 
    shuffle=True, 
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.eval_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True
)
controlnet = SD3ControlNetModel.from_pretrained("stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline(
        "stabilityai/stable-diffusion-3.5-large",
        controlnet=controlnet, 
        scheduler=noise_scheduler
    )
pipe = pipe.to(device)

sample_mask = train_dataset[0]['masks'].unsqueeze(0)
sample_image = train_dataset[0]['images'].unsqueeze(0)

sample_mask = sample_mask.to(device)
sample_image = sample_image.to(device)
noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)


optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config.learning_rate)

tokenizer = CLIPTokenizer.from_pretrained("clip-vocabulary")
text_encoder = CLIPTextModel.from_pretrained("clip-vocabulary").to(device)

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
     # Save the images
    test_dir = config.output_dir
    os.makedirs(test_dir, exist_ok=True)

    # run your diffusion pipeline, which returns a list of PIL Images:
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # pick the first one
    img = images[0]

    # save it as a single PNG per epoch
    img.save(f"{test_dir}/{epoch:04d}.png")

def validate(pipe, val_dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            clean_images = batch['images']
            clean_images = clean_images.to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noisy_images.to(device)

            # forward
            noise_pred = pipe.unet(
                    noisy_images, 
                    timesteps,
                    return_dict=False
                )[0]
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            bsz = noisy_images.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

    return total_loss / total_samples


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    patience  = 10        # how many epochs with no improvement before stopping
    best_val  = float("inf")
    counter   = 0
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
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        pipe.unet.train()

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            seg_masks = batch['masks']  # [B, 1, H, W]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                text_inputs = tokenizer(
                    ["a medical scan of an eye and a binary mask of the eye's vascular structure, disease severity dependent on tortuosity of blood vessels"],  # or any relevant prompt
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                encoder_hidden_states = text_encoder(**text_inputs).last_hidden_state
                down_res, mid_res = pipe.controlnet(
                        noisy_images, 
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=seg_masks
                    ).to_tuple()
                noise_pred = pipe.unet(
                    noisy_images, 
                    timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    encoder_hidden_states=encoder_hidden_states
                ).images[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

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
            
            val_loss = validate(pipe, val_dataloader)
            print(f"Epoch {epoch:2d} — val_loss: {val_loss:.4f}")
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipe)
            if val_loss < best_val:
                best_val = val_loss
                counter  = 0
                # save the “best” weights
                pipe.save_pretrained(config.output_dir)
            else:
                counter += 1
                print(f"  no improvement for {counter}/{patience} epochs")

            if counter >= patience or epoch == config.num_epochs:
                pipe.save_pretrained(config.output_dir)
                upload_folder(
                    repo_id=repo_id,
                    folder_path=config.output_dir,
                    commit_message=f"updated ROP weights",
                    ignore_patterns=["step_*", "epoch_*"],
                )    
                print(f"Early stopping at epoch {epoch} (val_loss plateaued)")
                break

                

args = (config, controlnet, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])