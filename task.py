import os
import celery
import uuid
from celery import Celery
from inference.stable_diffusion import generate_image_from_prompt, edit_image_with_prompt, edit_video_with_prompt