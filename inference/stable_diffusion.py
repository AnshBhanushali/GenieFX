import uuid
import os
import logging
import cv2
from PIL import Image

from .global_diffusion_pipelines import load_text2img_pipeline, load_img2img_pipeline

logger = logging.getLogger(__name__)


def generate_image_from_prompt(prompt: str, output_dir: str = ".", **kwargs) -> str:
    """
    Generates an image from text prompt using a globally loaded Stable Diffusion pipeline.
    """
    pipe = load_text2img_pipeline()  # calls the global pipeline loader
    num_inference_steps = kwargs.get("num_inference_steps", 30)
    guidance_scale = kwargs.get("guidance_scale", 7.5)

    logger.info(f"Generating image with prompt: {prompt}")
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]

    output_path = os.path.join(output_dir, f"output_{uuid.uuid4()}.png")
    image.save(output_path)
    logger.info(f"Image saved to {output_path}")
    return output_path


def edit_image_with_prompt(input_image_path: str, prompt: str, output_dir: str = ".", **kwargs) -> str:
    """
    Edits an existing image based on a text prompt using the globally loaded Img2Img pipeline.
    """
    pipe = load_img2img_pipeline()
    strength = kwargs.get("strength", 0.8)
    guidance_scale = kwargs.get("guidance_scale", 7.5)
    num_inference_steps = kwargs.get("num_inference_steps", 30)

    logger.info(f"Editing image {input_image_path} with prompt: {prompt}")

    init_bgr = cv2.imread(input_image_path)
    if init_bgr is None:
        raise FileNotFoundError(f"Input image not found at: {input_image_path}")

    init_rgb = cv2.cvtColor(init_bgr, cv2.COLOR_BGR2RGB)
    init_pil = Image.fromarray(init_rgb)

    image = pipe(
        prompt=prompt,
        image=init_pil,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    output_path = os.path.join(output_dir, f"edited_{uuid.uuid4()}.png")
    image.save(output_path)
    logger.info(f"Edited image saved to {output_path}")
    return output_path


def edit_video_with_prompt(input_video_path: str, prompt: str, output_dir: str = ".", **kwargs) -> str:
    """
    Edits a video frame-by-frame using the globally loaded Img2Img pipeline.
    """
    pipe = load_img2img_pipeline()
    strength = kwargs.get("strength", 0.5)
    guidance_scale = kwargs.get("guidance_scale", 7.5)
    num_inference_steps = kwargs.get("num_inference_steps", 30)
    keep_frames = kwargs.get("keep_frames", False)

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video not found: {input_video_path}")

    logger.info(f"Editing video {input_video_path} with prompt: {prompt}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{uuid.uuid4()}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    cap.release()

    # 2. Process each frame with Img2Img
    edited_frame_paths = []
    for fp in frame_paths:
        frame_img = cv2.imread(fp)
        if frame_img is None:
            continue
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        init_pil = Image.fromarray(frame_img_rgb)

        edited_pil = pipe(
            prompt=prompt,
            image=init_pil,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        edited_frame_path = os.path.join(output_dir, f"editedframe_{uuid.uuid4()}.png")
        edited_pil.save(edited_frame_path)
        edited_frame_paths.append(edited_frame_path)

    # 3. Re-assemble frames into video
    output_video_path = os.path.join(output_dir, f"editedvideo_{uuid.uuid4()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for edited_fp in edited_frame_paths:
        frame_img = cv2.imread(edited_fp)
        if frame_img is not None:
            out.write(frame_img)
    out.release()

    logger.info(f"Edited video saved to {output_video_path}")

    # Optional: Clean up frames
    if not keep_frames:
        for fp in frame_paths + edited_frame_paths:
            if os.path.exists(fp):
                os.remove(fp)

    return output_video_path
