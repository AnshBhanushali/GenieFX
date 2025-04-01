import os
import uuid
import logging
from typing import Optional

import torch
import cv2
import numpy as np
from PIL import Image

# Diffusers
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class DiffusionService:
    """
    Singleton service to handle Stable Diffusion pipelines.
    """
    _text2img_pipe = None
    _img2img_pipe = None

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id

    def _load_text2img_pipeline(self) -> StableDiffusionPipeline:
        if DiffusionService._text2img_pipe is None:
            logging.info("Loading text2img pipeline...")
            DiffusionService._text2img_pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                DiffusionService._text2img_pipe.to("cuda")
            logging.info("Text2Img pipeline loaded.")
        return DiffusionService._text2img_pipe

    def _load_img2img_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        if DiffusionService._img2img_pipe is None:
            logging.info("Loading img2img pipeline...")
            DiffusionService._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                DiffusionService._img2img_pipe.to("cuda")
            logging.info("Img2Img pipeline loaded.")
        return DiffusionService._img2img_pipe

    def generate_image_from_prompt(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        output_dir: str = "."
    ) -> str:
        """
        Generates an image from text prompt using a Stable Diffusion text-to-image pipeline.
        """
        pipe = self._load_text2img_pipeline()

        try:
            logging.info(f"Generating image from prompt: '{prompt}'")
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            raise

        filename = f"output_{uuid.uuid4()}.png"
        output_path = os.path.join(output_dir, filename)
        result.save(output_path)
        logging.info(f"Image saved to {output_path}")
        return output_path

    def edit_image_with_prompt(
        self,
        input_image_path: str,
        prompt: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        output_dir: str = "."
    ) -> str:
        """
        Edits an existing image based on the given text prompt using a Stable Diffusion img2img pipeline.
        """
        pipe = self._load_img2img_pipeline()

        try:
            init_image_bgr = cv2.imread(input_image_path)
            if init_image_bgr is None:
                raise ValueError(f"Could not read image file {input_image_path}")
            init_image_rgb = cv2.cvtColor(init_image_bgr, cv2.COLOR_BGR2RGB)
            init_pil = Image.fromarray(init_image_rgb)

            logging.info(f"Editing image {input_image_path} with prompt '{prompt}'")

            result = pipe(
                prompt=prompt,
                image=init_pil,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]

        except Exception as e:
            logging.error(f"Error editing image: {e}")
            raise

        filename = f"edited_{uuid.uuid4()}.png"
        output_path = os.path.join(output_dir, filename)
        result.save(output_path)
        logging.info(f"Edited image saved to {output_path}")
        return output_path

    def edit_video_with_prompt(
        self,
        input_video_path: str,
        prompt: str,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        output_dir: str = ".",
        keep_frames: bool = False
    ) -> str:
        """
        Edits each frame of an input video based on the text prompt using the img2img pipeline, then
        re-assembles the frames into a new video.
        """
        pipe = self._load_img2img_pipeline()
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video not found: {input_video_path}")

        # 1. Extract frames
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_paths = []
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = f"frame_{uuid.uuid4()}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_idx += 1
            cap.release()
        except Exception as e:
            logging.error(f"Error extracting frames from video: {e}")
            cap.release()
            raise

        # 2. Process each frame with Img2Img
        edited_frame_paths = []
        for fp in frame_paths:
            try:
                frame_img = cv2.imread(fp)
                frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                init_pil = Image.fromarray(frame_img_rgb)

                edited_pil = pipe(
                    prompt=prompt,
                    image=init_pil,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]

                edited_frame_filename = f"editedframe_{uuid.uuid4()}.png"
                edited_frame_path = os.path.join(output_dir, edited_frame_filename)
                edited_pil.save(edited_frame_path)
                edited_frame_paths.append(edited_frame_path)
            except Exception as e:
                logging.error(f"Error editing frame {fp}: {e}")
                # Optional: Could either skip or re-raise
                raise

        # 3. Re-assemble frames into video
        edited_video_filename = f"editedvideo_{uuid.uuid4()}.mp4"
        output_video_path = os.path.join(output_dir, edited_video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for edited_fp in edited_frame_paths:
            frame_img = cv2.imread(edited_fp)
            out.write(frame_img)
        out.release()

        logging.info(f"Edited video saved to {output_video_path}")

        # Optional: Clean up intermediate frames
        if not keep_frames:
            for fp in frame_paths + edited_frame_paths:
                if os.path.exists(fp):
                    os.remove(fp)
                    logging.debug(f"Removed {fp}")

        return output_video_path



