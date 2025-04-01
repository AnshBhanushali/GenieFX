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

def edit_video_with_prompt(input_video_path: str, prompt: str) -> str:
    # 1. Extract frames
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_paths = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f"frame_{uuid.uuid4()}.png"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1
    cap.release()

    # 2. Process each frame with Img2Img
    pipe = load_img2img_pipeline()

    output_frame_paths = []
    for fp in frame_paths:
        frame_img = cv2.imread(fp)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        init_pil = Image.fromarray(frame_img)
        
        edited_pil = pipe(
            prompt=prompt,
            image=init_pil,
            strength=0.5,
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]

        edited_frame_path = f"editedframe_{uuid.uuid4()}.png"
        edited_pil.save(edited_frame_path)
        output_frame_paths.append(edited_frame_path)

    # 3. Re-assemble frames into video
    output_video_path = f"editedvideo_{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = cv2.imread(output_frame_paths[0]).shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for edited_fp in output_frame_paths:
        frame_img = cv2.imread(edited_fp)
        out.write(frame_img)
    out.release()


    return output_video_path