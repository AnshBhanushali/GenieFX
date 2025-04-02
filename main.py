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