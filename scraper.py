import logging
import requests
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    )
}

def scrape_images_from_url(url: str, timeout: int = 10) -> List[str]:
    """
    Fetches the HTML of a page and extracts <img> tags with absolute URLs.
    For robust/dynamic scraping, consider using Selenium, Scrapy, or Playwright.

    :param url: The target URL to scrape.
    :param timeout: How long (in seconds) to wait for the server to send data.
    :return: A list of image URLs found in <img> tags.
    """
    image_links = []
    with requests.Session() as session:
        session.headers.update(DEFAULT_HEADERS)
        try:
            logger.info(f"Requesting URL: {url}")
            response = session.get(url, timeout=timeout)
            response.raise_for_status()  # raise HTTPError for bad responses
        except (RequestException, Timeout) as e:
            logger.error(f"Request failed for {url}: {e}")
            return image_links  # return empty list on failure

    # If request is successful, parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")

    for tag in img_tags:
        src = tag.get("src")
        if not src:
            continue

        if src.startswith("//"):
            src = f"https:{src}"
        if src.startswith("http://") or src.startswith("https://"):
            image_links.append(src)

    logger.info(f"Found {len(image_links)} image URLs in {url}")
    return image_links