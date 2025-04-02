import os
import logging
from celery import Celery

logger = logging.getLogger(__name__)

def make_celery_app() -> Celery:
    """
    Creates and configures a Celery application instance.
    """
    celery_app = Celery(
        "my_celery_app",
        broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
    )

    celery_app.conf.update(
        worker_concurrency=4,              # Adjust concurrency based on your hardware
        task_time_limit=600,              # Kill tasks that exceed 10 minutes
        task_soft_time_limit=550,         # Give a soft time limit to allow graceful shutdown
        broker_transport_options={
            'visibility_timeout': 3600
        },
        result_expires=3600,              # Results expire after an hour
        accept_content=['json'],          # Restrict accepted content to JSON
        task_serializer='json',
        result_serializer='json',
        enable_utc=True
    )

    return celery_app

celery_app = make_celery_app()