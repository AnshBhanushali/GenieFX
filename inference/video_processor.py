import os
import uuid
import cv2
import shutil
import torch
import numpy as np
from typing import Optional, Tuple, List

# Diffusers for Stable Diffusion
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# RIFE for interpolation
from torch_rife import RIFE

class VideoProcessor:
    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "video_temp",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        :param sd_model_id: Hugging Face model ID or local path for Img2Img pipeline.
        :param output_dir: Directory for temporary frame storage.
        :param device: 'cuda' or 'cpu'.
        """
        self.sd_model_id = sd_model_id
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)

        # (1) Load the Stable Diffusion Img2Img pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipe.to(self.device)

        # (2) Load the RIFE interpolation model
        # You can specify different model versions, e.g. "rife-v4", "rife-v4.6"
        self.rife_model = RIFE(model="rife-v4")
        self.rife_model.to(self.device)
        self.rife_model.eval()

    def extract_frames(self, video_path: str) -> Tuple[List[str], float]:
        """
        Extract frames from the input video into self.output_dir.
        Returns a list of frame paths and the video's FPS.
        """
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

    def _frame_to_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy BGR image to a normalized PyTorch tensor [B, C, H, W] on self.device.
        RIFE expects values in [0..1], in RGB order.
        """
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # [0..255] -> [0..1], float32
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        # Shape [H, W, C] -> [C, H, W], add batch dimension -> [1, C, H, W]
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor [B, C, H, W] in [0..1] range back to a NumPy BGR image.
        """
        # Remove batch dimension => shape [C, H, W]
        tensor = tensor.squeeze(0)
        # [C, H, W] -> [H, W, C]
        frame_rgb = tensor.permute(1, 2, 0).cpu().numpy()
        # [0..1] -> [0..255], float32 -> uint8
        frame_rgb = (frame_rgb * 255.0).clip(0, 255).astype(np.uint8)
        # Convert RGB -> BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def interpolate_frames(
        self,
        frame_paths: list,
        interpolation_factor: int = 2,
        interpolation_model: Optional[object] = None
    ) -> list:
        """
        Interpolate new frames between consecutive frames in frame_paths.
        If interpolation_factor=2, adds 1 new frame (t=0.5) per pair.
        If interpolation_factor=4, adds 3 new frames (t=0.25, 0.5, 0.75), etc.
        """
        if not interpolation_model:
            print("[video_processor] No interpolation model provided; skipping interpolation.")
            return frame_paths

        new_frame_paths = []

        for i in range(len(frame_paths) - 1):
            current_fp = frame_paths[i]
            next_fp = frame_paths[i + 1]

            # Always keep the current frame
            new_frame_paths.append(current_fp)

            # Load frames from disk
            frame_current_bgr = cv2.imread(current_fp)
            frame_next_bgr = cv2.imread(next_fp)
            if frame_current_bgr is None or frame_next_bgr is None:
                continue

            # Convert to Torch tensors
            current_tensor = self._frame_to_tensor(frame_current_bgr)
            next_tensor = self._frame_to_tensor(frame_next_bgr)

            # We'll generate (interpolation_factor - 1) new frames
            steps = interpolation_factor - 1
            for step_idx in range(1, steps + 1):
                # t = fraction of time between frames
                t = step_idx / (steps + 1)

                # RIFE interpolation
                inbetween_tensor = interpolation_model.interpolate(
                    current_tensor, next_tensor, timestep=t
                )

                # Convert to BGR image
                inbetween_bgr = self._tensor_to_frame(inbetween_tensor)

                # Save the new frame
                new_fp = os.path.join(self.output_dir, f"interpolated_{i}_{step_idx}.png")
                cv2.imwrite(new_fp, inbetween_bgr)
                new_frame_paths.append(new_fp)

        # Finally, add the last frame
        new_frame_paths.append(frame_paths[-1])
        return new_frame_paths

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

        # Read the first frame to get size
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


def process_video(
    input_video_path: str,
    prompt: str,
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    interpolation_factor: int = 1,
    remove_temp: bool = True
) -> str:
    """
    High-level function that orchestrates:
      1. Extract frames
      2. (Optionally) interpolate them via RIFE
      3. Run SD Img2Img on each frame
      4. Reassemble to final video
      5. (Optionally) cleanup temporary files
    """
    vp = VideoProcessor(sd_model_id=sd_model_id, output_dir=f"temp_{uuid.uuid4()}")
    frame_paths, fps = vp.extract_frames(input_video_path)

    # Interpolate frames if requested
    if interpolation_factor > 1:
        frame_paths = vp.interpolate_frames(
            frame_paths=frame_paths,
            interpolation_factor=interpolation_factor,
            interpolation_model=vp.rife_model  # pass the loaded RIFE model
        )

    # Apply Stable Diffusion to each frame
    edited_frame_paths = vp.process_frames_with_sd(
        frame_paths,
        prompt=prompt,
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=20
    )

    # Reassemble
    output_video_path = f"editedvideo_{uuid.uuid4()}.mp4"
    vp.reassemble_video(edited_frame_paths, output_video_path, fps=fps)

    # Clean up
    if remove_temp:
        vp.cleanup()

    return output_video_path
