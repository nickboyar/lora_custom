from pathlib import Path
import os
import torch
from diffusers import DiffusionPipeline
from hydra import compose, initialize
from accelerate import Accelerator
from loguru import logger
from accelerate.utils import ProjectConfiguration, set_seed

BASE_PATH = Path(__file__).parent.parent

def infer():
    logger.info("Starting generation")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer.yaml")

    lora_dir = str(BASE_PATH / cfg.all_lora_dir / cfg.lora_dir)
    output_dir = str(BASE_PATH / cfg.generate_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Directory '{output_dir}' created or already exists.")
    except Exception as e:
        logger.error(f"Error creating directory: {e}")

    accelerator_project_config = ProjectConfiguration(project_dir=lora_dir)    
    accelerator = Accelerator(
        project_config=accelerator_project_config,
    )
    weight_dtype = torch.float32    
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )

    if cfg.apply_lora:
        pipeline.load_lora_weights(lora_dir)
        logger.info(f"Lora weights downwloaded successfully")

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator = generator.manual_seed(cfg.seed)

    for ind in range(cfg.num_img):
        try:
            image = pipeline(cfg.prompt,
                                num_inference_steps=cfg.num_inference_steps, 
                                guidance_scale=cfg.guidance_scale,
                                generator=generator).images[0]
            image.save(f"{output_dir}/{ind + 1}.png")
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
