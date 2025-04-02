import os
import celery
import uuid
from celery import Celery
from inference.stable_diffusion import generate_image_from_prompt, edit_image_with_prompt, edit_video_with_prompt

celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)
