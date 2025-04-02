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

@celery_app.task(name="process_text_to_image_task")
def process_text_to_image_task(prompt: str):
    # Generate the image
    output_path = generate_image_from_prompt(prompt)
    return {"output_path": output_path}

@celery_app.task(name="process_image_edit_task")
def process_image_edit_task(prompt: str, input_file_path: str):
    output_path = edit_image_with_prompt(input_file_path, prompt)
    # Clean up input file if needed
    return {"output_path": output_path}

@celery_app.task(name="process_video_edit_task")
def process_video_edit_task(prompt: str, input_file_path: str):
    output_video_path = edit_video_with_prompt(input_file_path, prompt)
    # Clean up input file if needed
    return {"output_path": output_video_path}