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
