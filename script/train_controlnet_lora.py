#!/usr/bin/env python
# coding=utf-8
"""
Modified ControlNet LoRA Training Script.
Adapted for Car Damage Inpainting using YOLO/SAM masks.
FIXED: Uses peft.get_peft_model() to wrap ControlNet for LoRA compatibility.
FIXED: Robust recursive dataset loading.
FIXED: Ensures conditioning image is 3-channel RGB to prevent RuntimeError.
FIXED: Added accelerator.autocast() to fix mixed dtype errors.
FIXED: Added missing checkpoint resume logic.
"""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import glob 

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    DiffusionPipeline, 
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel, 
    ControlNetModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from PIL import Image
from torch.utils.data import Dataset

if is_wandb_available():
    import wandb

check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class SimpleControlNetDataset(Dataset):
    """
    Robust dataset loader that recursively finds images and pairs them with masks.
    """
    def __init__(self, data_root, size=512):
        self.data_root = data_root
        self.size = size
        
        if not os.path.exists(data_root):
            raise ValueError(f"Data root directory does not exist: {data_root}")
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_paths = []
        
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_root, '**', ext), recursive=True))
            
        self.image_paths = [
            p for p in self.image_paths 
            if "mask" not in os.path.basename(p).lower() 
            and "label" not in os.path.basename(p).lower()
        ]
        
        if len(self.image_paths) == 0:
            print(f"\nERROR: No images found in {data_root}")
            # Debug print logic removed for brevity in this final version but retained in user's file.
            raise ValueError(f"No training images found in {data_root}. Please check your dataset path.")

        logger.info(f"Found {len(self.image_paths)} training images in {data_root}")
        
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            original_image = Image.new("RGB", (self.size, self.size))

        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        
        possible_masks = [
            os.path.join(self.data_root, "masks", basename + ".png"),
            os.path.join(self.data_root, "masks", basename + ".jpg"),
        ]
        
        mask_path = None
        for p in possible_masks:
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            conditioning_image = Image.open(mask_path).convert("RGB") 
        else:
            conditioning_image = Image.new("RGB", original_image.size, (0, 0, 0))

        return {
            "pixel_values": self.transform_image(original_image),
            "conditioning_pixel_values": self.transform_mask(conditioning_image),
            "caption": "a high quality photo of a perfectly repaired car"
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )

    # Arguments added for optimization
    parser.add_argument( 
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers memory efficient attention." 
    ) 

    parser.add_argument( 
        "--set_grads_to_none", 
        action="store_true", 
        default=True,
        help="Setting gradients to None is a more efficient approach than setting them to zero.", 
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    
    # --- ARGS FOR DATASET AND LORA FLAGS ---
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the folder containing your training images.",
    )
    parser.add_argument(
        "--lora_controlnet_only",
        action="store_true",
        help="If set, only trains the ControlNet weights using LoRA.",
    )
    
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500, 
        help="This is the old save steps argument, now primarily controlled by --checkpointing_steps"
    ) 
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Setup Accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load Scheduler, Tokenizer, UNet, VAE, Text Encoder
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # --- CONTROLNET INITIALIZATION ---
    logger.info("Initializing ControlNet weights from UNet...")
    controlnet = ControlNetModel.from_unet(unet)

    # Freeze Base Models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False) # Freeze base ControlNet weights

    # --- LORA SETUP (The Fix) ---
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    # *** FIX: Use get_peft_model wrapper to apply LoRA to ControlNet ***
    controlnet = get_peft_model(controlnet, lora_config)
    
    # Log the trainable parameters
    logger.info(controlnet.print_trainable_parameters())

    # Set mixed precision for the models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # ControlNet will be moved to device by Accelerator.prepare, but ensure dtype is right
    controlnet.to(accelerator.device) 
    

    # Enable Gradient Checkpointing
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Optimizer
    # The peft wrapper ensures only LoRA layers require gradients
    params_to_optimize = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- DATASET & DATALOADER ---
    train_dataset = SimpleControlNetDataset(data_root=args.train_data_dir, size=args.resolution)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        # Tokenize captions
        captions = [example["caption"] for example in examples]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids
        
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with Accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # --- NEW CHECKPOINT RESUME LOGIC ---
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            accelerator.print(f"Resuming training from step {initial_global_step}")
    else:
        initial_global_step = 0


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            controlnet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly.")
            
    # --- TRAINING LOOP ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = initial_global_step
    
    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps), 
        initial=initial_global_step, 
        desc="Steps", 
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(args.num_train_epochs):
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
            if accelerator.sync_gradients and global_step >= args.max_train_steps:
                break
                
            with accelerator.accumulate(controlnet):
                # 1. Convert images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 3. Sample timesteps
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 4. Add noise (Forward Diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 5. Get Text Embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                
                # 6. Get ControlNet Condition (Mask)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # *** FIX: Added autocast context for mixed precision stability ***
                with accelerator.autocast():
                    # 7. ControlNet Forward Pass
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    # 8. UNet Forward Pass (Conditioned by ControlNet)
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # 9. Compute Loss
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 10. Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Save Checkpoint
                # Use args.checkpointing_steps for save frequency
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save Final LoRA Weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Cast back to float32 before saving the LoRA weights
        controlnet = controlnet.to(torch.float32)
        
        # Save only the LoRA adapters
        controlnet_lora_state_dict = get_peft_model_state_dict(controlnet)
        
        # Save weights in the diffusers format compatible with ControlNet
        StableDiffusionControlNetPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=controlnet_lora_state_dict, 
            safe_serialization=True,
        )
        logger.info(f"Model saved to {args.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()