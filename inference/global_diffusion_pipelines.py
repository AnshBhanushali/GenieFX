# inference/global_diffusion_pipelines.py
import logging
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

logger = logging.getLogger(__name__)

class GlobalPipelines:
    """
    A singleton-like class to hold globally loaded pipelines
    so that Celery tasks do not have to reload them every time.
    """
    text2img_pipe = None
    img2img_pipe = None

def load_text2img_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    if GlobalPipelines.text2img_pipe is None:
        logger.info("Loading StableDiffusionPipeline (text2img) globally...")
        GlobalPipelines.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            GlobalPipelines.text2img_pipe.to("cuda")
    return GlobalPipelines.text2img_pipe

def load_img2img_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    if GlobalPipelines.img2img_pipe is None:
        logger.info("Loading StableDiffusionImg2ImgPipeline globally...")
        GlobalPipelines.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            GlobalPipelines.img2img_pipe.to("cuda")
    return GlobalPipelines.img2img_pipe
