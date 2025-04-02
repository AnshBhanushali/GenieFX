import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from celery import Celery
from pydantic import BaseModel
import uuid
import uvicorn

app = FastAPI(
    title="AI Media Generation & Editing API",
    description="Endpoints for image/video generation/editing",
    version="1.0.0",
)

# Celery config (same as in tasks.py)
celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

class TextPrompt(BaseModel):
    prompt: str

@app.post("/scrape_images")
def scrape_images(url: str):
    """
    Example endpoint to scrape images from a URL.
    """
    image_links = scrape_images_from_url(url)
    return {"images_found": image_links}

@app.post("/text-to-image")
def text_to_image(prompt: TextPrompt):
    """
    Kick off a Celery task for text-to-image generation.
    Returns a task_id that client can poll for result.
    """
    task = process_text_to_image_task.delay(prompt.prompt)
    return {"task_id": task.id}

@app.post("/image-edit")
def image_edit(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Kick off a Celery task for image editing.
    """
    # Save the uploaded image temporarily
    input_file_path = f"temp_{uuid.uuid4()}.png"
    with open(input_file_path, "wb") as f:
        f.write(file.file.read())

    task = process_image_edit_task.delay(prompt, input_file_path)
    return {"task_id": task.id}

@app.post("/video-edit")
def video_edit(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Kick off a Celery task for video editing.
    """
    input_file_path = f"temp_{uuid.uuid4()}.mp4"
    with open(input_file_path, "wb") as f:
        f.write(file.file.read())

    task = process_video_edit_task.delay(prompt, input_file_path)
    return {"task_id": task.id}

