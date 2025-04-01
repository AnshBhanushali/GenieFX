import os
import uuid
import cv2
import shutil
import torch
import numpy as np
from typing import Optional


from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

class VideoProcessor:
    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "video_temp",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.sd_model_id = sd_model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load your Stable Diffusion pipeline (img2img)
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.pipe.to(device)

    def extract_frames(self, video_path: str) -> (list, float):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_paths = []

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_file = os.path.join(self.output_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(frame_file, frame)
            frame_paths.append(frame_file)
            frame_index += 1

        cap.release()
        return frame_paths, fps
    
    def process_frames_with_sd(
        self,
        frame_paths: list,
        prompt: str,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
    ) -> list:
        """
        Pass frames one-by-one through Stable Diffusion Img2Img.
        Returns new frame paths with edited images.
        """
        edited_frame_paths = []
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            init_pil = Image.fromarray(frame_rgb)

            edited = self.pipe(
                prompt=prompt,
                image=init_pil,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]

            out_path = os.path.join(self.output_dir, f"edited_{i:06d}.png")
            edited.save(out_path)
            edited_frame_paths.append(out_path)

        return edited_frame_paths

    def reassemble_video(
        self,
        frame_paths: list,
        output_path: str,
        fps: float
    ) -> str:
        """
        Rebuild the video from the processed frames using OpenCV’s VideoWriter.
        """
        if not frame_paths:
            raise ValueError("[video_processor] No frames to assemble.")

        # Read first frame to get size
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for fp in frame_paths:
            frame = cv2.imread(fp)
            out.write(frame)

        out.release()
        return output_path

    def cleanup(self):
        """
        Optional: remove the temporary directory if desired.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)