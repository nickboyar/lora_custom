import os
from fastapi import FastAPI, UploadFile, Form, File
import zipfile
import io
import yaml
from loguru import logger
import uvicorn
from train import train
from infer import infer
from utils import get_loras_dirs, get_image_paths

app = FastAPI()

@app.post("/upload_images")
async def upload_images_zip(zip_file: UploadFile):
    """
    Uploads a ZIP file and processes its contents.
    """
    logger.info("Starting load zip")
    if zip_file.content_type != "application/zip":
        return {"message": "Invalid file type. Please upload a ZIP file."}
    try:
        zip_content = await zip_file.read()
        zip_buffer = io.BytesIO(zip_content)
        with zipfile.ZipFile(zip_buffer, 'r') as temp_zip:
            temp_zip.extractall(path='../tmp/train')
        return {"filename": zip_file.filename, "message": "ZIP file uploaded and processed successfully."}
    except zipfile.BadZipFile:
        return {"filename": zip_file.filename, "message": "Invalid ZIP file."}
    except Exception as e:
        return {"filename": zip_file.filename, "message": f"An error occurred: {e}"}

@app.post("/upload_train_cfg")
async def upload_train_cfg(train_cfg_file: UploadFile):
    """
    Uploads a train cfg file and processes its contents.
    """
    logger.info("Starting load train cfg")
    if train_cfg_file.content_type != "application/x-yaml" and train_cfg_file.content_type != "text/yaml":
        return {"message": "Invalid file type. Please upload a YAML file."}
    try:
        content = await train_cfg_file.read()
        save_path = os.path.join('../configs', train_cfg_file.filename)
        with open(save_path, "wb") as f:
            f.write(content)
        return {"filename": train_cfg_file.filename, "content_type": train_cfg_file.content_type}
    except yaml.YAMLError as e:
        return {"message": f"Error parsing YAML: {e}"}
    except Exception as e:
        return {"message": f"An error occurred: {e}"}

@app.post("/upload_generate_cfg")
async def upload_generate_cfg(infer_cfg_file: UploadFile):
    """
    Uploads a train cfg file and processes its contents.
    """
    logger.info("Starting load generate cfg")
    if infer_cfg_file.content_type != "application/x-yaml" and infer_cfg_file.content_type != "text/yaml":
        return {"message": "Invalid file type. Please upload a YAML file."}
    try:
        content = await infer_cfg_file.read()
        save_path = os.path.join('../configs', infer_cfg_file.filename)
        with open(save_path, "wb") as f:
            f.write(content)
        return {"filename": infer_cfg_file.filename, "content_type": infer_cfg_file.content_type}
    except yaml.YAMLError as e:
        return {"message": f"Error parsing YAML: {e}"}
    except Exception as e:
        return {"message": f"An error occurred: {e}"}

@app.post("/train")
def train_lora():
    """
    Train lora with user cfg and data.
    """
    logger.info("Starting train lora")
    try:
        train()
    except Exception as e:
        return {"message": f"An error occurred: {e}"}
    return {"message": "Training finished successfully."}

@app.post("/generate")
async def generate_image():
    """
    Generate output in output_classic dir
    """
    logger.info("Starting generate images")
    try:
        infer()
    except Exception as e:
        return {"message": f"An error occurred: {e}"}
    return {"message": "Image generated successfully."}


@app.get("/generated_images")
def get_generated_images():
    """
    Get list of all generated images
    """
    logger.info("Get list images")
    try:
        loras = get_image_paths()
    except Exception as e:
        return {"message": f"An error occurred: {e}"}
    return {"images": loras}

@app.get("/list_loras")
def get_list_loras():
    """
    Get list of all available LoRAs
    """
    logger.info("Get list loras")
    try:
        loras = get_loras_dirs()
    except Exception as e:
        return {"message": f"An error occurred: {e}"}
    return {"loras": loras}

@app.get("/health")
def health():
    logger.info("check")
    return {"status": "ok"}

if __name__=='__main__':
   uvicorn.run('app:app', host="127.0.0.1", port=8081, log_level="info", reload=True)
