#!/usr/bin/env python
# coding=utf-8
"""
Modified ControlNet LoRA Training Script.
Adapted for Car Damage Inpainting using YOLO/SAM masks.
"""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    DiffusionPipeline, 
    StableDiffusionControlNetPipeline, # Changed pipeline
    UNet2DConditionModel, 
    ControlNetModel # Added ControlNetModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# *** CUSTOM IMPORTS FOR DATASET ***
from PIL import Image
from torch.utils.data import Dataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# --- CUSTOM DATASET CLASS INTEGRATED DIRECTLY ---
class SimpleControlNetDataset(Dataset):
    """
    A simple dataset that loads images and masks from directories.
    Assumes you have run your YOLO/SAM generation and saved files to disk.
    """
    def __init__(self, data_root, size=512):
        self.data_root = data_root
        self.size = size
        self.image_files = [f for f in os.listdir(data_root) if f.endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in f]
        
        # Image transforms
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Mask transforms (No normalization, just resize and tensor)
        self.transform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.data_root, img_name)
        
        # Assume mask has same name but maybe different extension or suffix if you organized it that way
        # Ideally, save your masks as "filename_mask.png" in the same folder or a 'masks' subfolder
        # EDIT THIS LOGIC TO MATCH YOUR SAVED DATA STRUCTURE
        mask_name = os.path.splitext(img_name)[0] + ".png" # Assuming masks are png
        # If you saved masks in the same folder with the same name (not recommended as it overwrites):
        # Let's assume you have a 'masks' subfolder for this script to work out of the box
        # OR you used your YOLO/SAM code to save pairs. 
        
        # Placeholder logic: Loading the image as both source and target for now
        # You MUST replace this with your actual mask loading logic
        original_image = Image.open(image_path).convert("RGB")
        
        # TRY to find the mask. If you haven't generated them yet, this will fail.
        # This script expects you to have a 'masks' folder inside your data_root
        mask_path = os.path.join(self.data_root, "masks", mask_name)
        
        if os.path.exists(mask_path):
            conditioning_image = Image.open(mask_path).convert("L")
        else:
            # Fallback: create a blank mask (for testing only)
            conditioning_image = Image.new("L", original_image.size, 0)

        return {
            "pixel_values": self.transform_image(original_image),
            "conditioning_pixel_values": self.transform_mask(conditioning_image),
            "caption": "a high quality photo of a car" # Placeholder caption
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
        help="Whether training should be resumed from a previous checkpoint.",
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

    # Inserting this new parser argument around other optimization settings:
    parser.add_argument( 
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers memory efficient attention." 
    ) 

    parser.add_argument( 
        "--set_grads_to_none", 
        action="store_true", 
        default=True,
        help=(
            "Setting gradients to None is a more efficient approach than setting them to zero. "
            "It's recommended when using gradient accumulation."
        ), 
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
    
    # Unused but kept for compatibility with launch scripts that might pass them
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500) # Added to fix your error
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
    # We initialize ControlNet from the UNet
    logger.info("Initializing ControlNet weights from UNet...")
    controlnet = ControlNetModel.from_unet(unet)

    # Freeze Base Models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False) # Freeze base ControlNet weights

    # --- LORA SETUP ---
    # We apply LoRA *only* to the ControlNet
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    # Add LoRA adapter to ControlNet
    controlnet.add_adapter(lora_config)
    
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
    controlnet.to(accelerator.device, dtype=weight_dtype)

    # Make sure trainable LoRA params are float32
    cast_training_params(controlnet, dtype=torch.float32)

    # Enable Gradient Checkpointing
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Optimizer
    params_to_optimize = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- DATASET & DATALOADER ---
    # Instantiating the custom dataset class defined at the top
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


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            # Note: We skip the version check here as we are using the latest source install
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

    global_step = 0
    
    # Progress bar
    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
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

                # 7. ControlNet Forward Pass
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # 8. UNet Forward Pass (Conditioned by ControlNet)
                # Predict the noise residual
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
        unet = unet.to(torch.float32)
        controlnet = controlnet.to(torch.float32)
        
        # Save only the LoRA adapters
        controlnet_lora_state_dict = get_peft_model_state_dict(controlnet)
        
        StableDiffusionControlNetPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=controlnet_lora_state_dict, # Saving as UNet compatible layers
            safe_serialization=True,
        )
        logger.info(f"Model saved to {args.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()