import random
from pathlib import Path
from loguru import logger

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from hydra import compose, initialize

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module

from utils import make_dataset

BASE_PATH = Path(__file__).parent.parent
TMP_PATH = "tmp/train"

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def tokenize_captions(examples, is_train=True):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml")
    captions = []
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    for caption in examples['text']:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{'text'}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def preprocess_train(examples):

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml")

    interpolation = getattr(transforms.InterpolationMode, cfg.image.image_interpolation_mode, None)
    train_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.image.resolution, interpolation=interpolation),
            transforms.CenterCrop(cfg.image.resolution) if cfg.image.center_crop else transforms.RandomCrop(cfg.image.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [image.convert("RGB") for image in examples['image']]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

def train():
    
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml")

    lora_dir = BASE_PATH / cfg.all_lora_dir / cfg.lora_dir
    data_dir = BASE_PATH / TMP_PATH / cfg.data_dir

    accelerator_project_config = ProjectConfiguration(project_dir=lora_dir)    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    logger.info("Load scheduler, tokenizer and models.")
    
    unet_lora_config = LoraConfig(
        r=cfg.lora_config.rank,
        lora_alpha=cfg.lora_config.lora_alpha,
        init_lora_weights=cfg.lora_config.init_lora_weights,
        target_modules=cfg.lora_config.target_modules,
    )

    logger.info("Config lora.")
    
    weight_dtype = torch.float32
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    unet.add_adapter(unet_lora_config)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    unet.enable_gradient_checkpointing()
    torch.backends.cuda.matmul.allow_tf32 = True
    
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )

    dataset = make_dataset(data_dir)
    train_dataset = dataset["train"].with_transform(preprocess_train)

    logger.info("Make and preprocess dataset.")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = cfg.train.lr_warmup_steps * accelerator.num_processes
    num_training_steps_for_scheduler = cfg.train.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {cfg.num_train_epochs}")

    progress_bar = tqdm(
        range(0, cfg.num_train_epochs),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(cfg.num_train_epochs):
        unet.train()
        train_loss = 0.0
        print('epoch', epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    
            if accelerator.sync_gradients:
                global_step += 1
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= cfg.max_train_steps:
                break
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=lora_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )         

        logger.info(f"Saved state to {lora_dir}")
