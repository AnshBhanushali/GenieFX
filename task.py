import logging
from typing import Dict
from celery import Celery

from inference.stable_diffusion import (
    generate_image_from_prompt,
    edit_image_with_prompt,
    edit_video_with_prompt
)

logger = logging.getLogger(__name__)

# ✅ Use in-memory broker + result backend
celery_app = Celery(
    "task",
    broker="memory://",
    backend="rpc://"
)

celery_app.conf.task_track_started = True
celery_app.conf.result_expires = 3600  # 1 hour

@celery_app.task(name="process_text_to_image_task", bind=True, max_retries=3, default_retry_delay=60)
def process_text_to_image_task(self, prompt: str, **kwargs) -> Dict[str, str]:
    try:
        logger.info(f"Received text2img task with prompt: {prompt}")
        output_path = generate_image_from_prompt(prompt, **kwargs)
        return {"output_path": output_path}
    except Exception as exc:
        logger.error(f"Error in text2img task: {exc}")
        self.retry(exc=exc)

@celery_app.task(name="process_image_edit_task", bind=True, max_retries=3, default_retry_delay=60)
def process_image_edit_task(self, prompt: str, input_file_path: str, **kwargs) -> Dict[str, str]:
    try:
        logger.info(f"Received image edit task with prompt: {prompt}, file: {input_file_path}")
        output_path = edit_image_with_prompt(input_file_path, prompt, **kwargs)
        return {"output_path": output_path}
    except Exception as exc:
        logger.error(f"Error in image edit task: {exc}")
        self.retry(exc=exc)

@celery_app.task(name="process_video_edit_task", bind=True, max_retries=3, default_retry_delay=60)
def process_video_edit_task(self, prompt: str, input_file_path: str, **kwargs) -> Dict[str, str]:
    try:
        logger.info(f"Received video edit task with prompt: {prompt}, file: {input_file_path}")
        output_video_path = edit_video_with_prompt(input_file_path, prompt, **kwargs)
        return {"output_path": output_video_path}
    except Exception as exc:
        logger.error(f"Error in video edit task: {exc}")
        self.retry(exc=exc)
