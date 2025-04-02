import os
import uuid
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from celery import Celery

# 1. Import the scraper and Celery tasks
from backend.scraping.scraper import scrape_images_from_url
from backend.tasks import (
    process_text_to_image_task,
    process_image_edit_task,
    process_video_edit_task
)

# 2. Configure logging (adjust level as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3. Create a directory for temporary files
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# 4. Initialize FastAPI
app = FastAPI(
    title="AI Media Generation & Editing API",
    description="Endpoints for image/video generation/editing",
    version="1.0.0",
)

# 5. Initialize Celery (same broker and backend as tasks.py)
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
    Scrapes a given URL for image links using 'scrape_images_from_url'.
    Returns a JSON response with all found image URLs.
    """
    try:
        logger.info(f"Scraping images from: {url}")
        image_links = scrape_images_from_url(url)
        return {"images_found": image_links}
    except Exception as e:
        logger.error(f"Error scraping images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-image")
def text_to_image(prompt: TextPrompt):
    """
    Kicks off a Celery task for text-to-image generation.
    Returns a task_id that the client can poll for a result.
    """
    try:
        logger.info(f"Received text-to-image request with prompt: {prompt.prompt}")
        task = process_text_to_image_task.delay(prompt.prompt)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error starting text-to-image task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image-edit")
def image_edit(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Kicks off a Celery task for image editing using the uploaded file and a text prompt.
    Returns a task_id for polling results.
    """
    try:
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"temp_{uuid.uuid4()}{file_extension}"
        input_file_path = os.path.join(TEMP_DIR, file_name)

        # Save the uploaded image temporarily
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Image received. Saved to: {input_file_path}")
        task = process_image_edit_task.delay(prompt, input_file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in image edit request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video-edit")
def video_edit(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Kicks off a Celery task for video editing using the uploaded file and a text prompt.
    Returns a task_id for polling results.
    """
    try:
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"temp_{uuid.uuid4()}{file_extension}"
        input_file_path = os.path.join(TEMP_DIR, file_name)

        # Save the uploaded video temporarily
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Video received. Saved to: {input_file_path}")
        task = process_video_edit_task.delay(prompt, input_file_path)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error in video edit request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/{task_id}")
def get_result(task_id: str):
    """
    Polling endpoint. Retrieves the status/result of a Celery task by ID.
    Possible states: PENDING, STARTED, SUCCESS, FAILURE, or UNKNOWN.
    On SUCCESS, returns whatever the Celery task returned (e.g., a file path).
    """
    try:
        task_result = celery_app.AsyncResult(task_id)
        if task_result.state == "PENDING":
            return {"status": "PENDING"}
        elif task_result.state == "STARTED":
            return {"status": "STARTED"}
        elif task_result.state == "SUCCESS":
            return {"status": "SUCCESS", "result": task_result.result}
        elif task_result.state == "FAILURE":
            return {"status": "FAILURE", "error": str(task_result.result)}
        else:
            return {"status": "UNKNOWN"}
    except Exception as e:
        logger.error(f"Error fetching result for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the server with auto-reload for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
