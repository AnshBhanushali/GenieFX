# main.py

import os
import uuid
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from celery import Celery
import os

# Celery
from celery import Celery

# Scraping function
from scraper import scrape_images_from_url

# Celery tasks for text2img, image edit, video edit
from task import (
    process_text_to_image_task,
    process_image_edit_task,
    process_video_edit_task
)

# Optional: If you want to serve static files (images/videos) from the backend
# from fastapi.staticfiles import StaticFiles
# app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Media Generation & Editing API",
    description="Endpoints for image/video generation/editing with Celery tasks",
    version="1.0.0",
)

# Enable CORS so frontend at localhost:3000 can reach this API
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
    # Add your production domain here if deployed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary uploads folder
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

celery_app = Celery(
    "task",
    broker="memory://",
    backend="rpc://",
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
