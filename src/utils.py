import os, json
from pathlib import Path
from hydra import compose, initialize
from datasets import load_dataset

BASE_PATH = Path(__file__).parent.parent

def make_dataset(data_files):
    dataset = load_dataset(
        "imagefolder",
        data_dir=data_files,
        )
    return dataset

def get_loras_dirs():

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml")
    lora_dir = str(BASE_PATH / cfg.all_lora_dir)
    return os.listdir(lora_dir)

def get_image_paths():

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer.yaml")
    gen_dir = str(BASE_PATH / cfg.generate_dir)

    full_paths = [os.path.join(gen_dir, item) for item in os.listdir(gen_dir)]
    return full_paths

