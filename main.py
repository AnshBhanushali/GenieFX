import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from celery import Celery
from pydantic import BaseModel
import uuid
import uvicorn