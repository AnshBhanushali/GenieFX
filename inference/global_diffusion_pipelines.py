import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

def load_text2img_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    return pipe.to(device)

def load_img2img_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    return pipe.to(device)
