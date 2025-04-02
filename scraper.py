# backend/scraping/tasks.py

import logging
from celery import Celery
from typing import Dict, List

from backend.scraping.scraper import scrape_images_from_url

logger = logging.getLogger(__name__)

celery_app = Celery(
    "scraper_tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

@celery_app.task(name="scrape_images_task", bind=True, max_retries=3, default_retry_delay=60)
def scrape_images_task(self, url: str, timeout: int = 10) -> Dict[str, List[str]]:
    """
    Celery task that scrapes a given URL for image URLs.

    :param url: The URL to scrape.
    :param timeout: Request timeout in seconds.
    :return: Dictionary containing the list of image URLs found.
    """
    try:
        logger.info(f"Scraping images from URL: {url}")
        images = scrape_images_from_url(url, timeout=timeout)
        return {"images": images}
    except Exception as exc:
        logger.error(f"Error scraping {url}: {exc}")
        self.retry(exc=exc)
