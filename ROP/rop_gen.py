import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import random_split

from torchvision import transforms
import multiprocessing
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
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


login(token = "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")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'rop_gen_25'  # the model namy locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = True  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

config.dataset_name = "imagefolder"
dataset = load_dataset(config.dataset_name, data_dir="/media/yuganlab/blackstone/HZL/Data/synthetic")


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
# preprocess_mask = transforms.Compose([
#     transforms.Resize((config.image_size, config.image_size)),
#     transforms.ToTensor(),  # Keep as [0,1]
# ])

def transform(examples):
    images = [preprocess(image.convert("L")) for image in examples["image"]]
    # masks  = [preprocess_mask(mask.convert("L")) for mask in examples["mask"]]
    return {"images": images}



dataset.set_transform(transform)

full_train = dataset["train"]

val_frac = 0.1
n_val = int(len(full_train) * val_frac)
n_train = len(full_train) - n_val
train_dataset, val_dataset = random_split(full_train, [n_train, n_val])

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

model = UNet2DModel.from_pretrained(
    '/media/yuganlab/blackstone/rop_gen_25/unet', 
    use_safetensors = True
).to(device)
sample_image = dataset['train'][0]['images'].unsqueeze(0)
sample_image = sample_image.to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

noise = torch.randn(sample_image.shape, device=device)
timesteps = torch.LongTensor([50]).to(device)
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)



noise_pred = model(noisy_image, timesteps).sample.to(device)
loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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

def evaluate(config, epoch, pipeline):
 

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

def validate(model, val_dataloader):
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
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
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
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
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
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline = pipeline.to(device)
            
            val_loss = validate(model, val_dataloader)
            print(f"Epoch {epoch:2d} — val_loss: {val_loss:.4f}")
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
            if val_loss < best_val:
                best_val = val_loss
                counter  = 0
                # save the “best” weights
                pipeline.save_pretrained(config.output_dir)
            else:
                counter += 1
                print(f"  no improvement for {counter}/{patience} epochs")

            if counter >= patience or epoch == config.num_epochs:
                pipeline.save_pretrained(config.output_dir)
                upload_folder(
                    repo_id=repo_id,
                    folder_path=config.output_dir,
                    commit_message=f"updated ROP weights",
                    ignore_patterns=["step_*", "epoch_*"],
                )    
                print(f"Early stopping at epoch {epoch} (val_loss plateaued)")
                break

                

args = (config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])