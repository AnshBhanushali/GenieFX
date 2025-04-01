import torch
import os
import uuid
import cv2
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

def load_text2img_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe

def generate_image_from_prompt(prompt: str) -> str:
    pipe = load_text2img_pipeline()
    image = pipe(prompt, num_inference_steps=30).images[0]
    output_path = f"output_{uuid.uuid4()}.png"
    image.save(output_path)
    return output_path

def load_img2img_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe

def edit_image_with_prompt(input_image_path: str, prompt: str) -> str:
    pipe = load_img2img_pipeline()
    init_image = cv2.imread(input_image_path)
    init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL
    from PIL import Image
    init_pil = Image.fromarray(init_image)
    
    strength = 0.8 
    guidance_scale = 7.5
    
    image = pipe(
        prompt=prompt,
        image=init_pil,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=30
    ).images[0]

    output_path = f"edited_{uuid.uuid4()}.png"
    image.save(output_path)
    return output_path