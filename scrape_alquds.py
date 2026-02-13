# scrape_alquds.py
#
# Scrapes selected Al-Quds University pages (admissions, scholarships,
# housing, faculties, etc.), cleans the HTML to plain text, and saves
# them into data/scraped/*.txt so the chatbot can use them.

import os
import time
import re
import urllib.parse
from collections import deque

import requests
from bs4 import BeautifulSoup

# -----------------------
# PATHS
# -----------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data", "scraped")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------
# CONFIG – EDIT THESE
# -----------------------

# You can use /en/ or /ar/ depending on what you want.
# Add or remove URLs as needed.
START_URLS = [
   "https://www.alquds.edu/en/admissions-registration/",
    "https://www.alquds.edu/en/aid-scholarships/",
    "https://ulife.alquds.edu/our_services/housing/",
    "https://ulife.alquds.edu/our_services/campus-facilities/",
    "https://www.alquds.edu/en/admissions-registration/admissions/bachelors-program/offered-programs/",
]

ALLOWED_DOMAIN = "www.alquds.edu"
MAX_PAGES = 80          # safety cap
REQUEST_DELAY = 1.0     # seconds between requests (be polite to server)

# Only follow links whose path contains one of these segments
ALLOWED_PATH_SNIPPETS = [
    "admissions",
    "admission",
    "scholar",
    "housing",
    "residence",
    "faculty",
    "faculties",
    "registration",
]

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def normalize_url(url: str) -> str:
    """Strip fragments and trailing slash so URLs are comparable."""
    url = url.split("#")[0]
    if url.endswith("/"):
        url = url[:-1]
    return url


def should_follow(href: str) -> bool:
    """Decide whether to follow a link based on domain + path."""
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc and parsed.netloc != ALLOWED_DOMAIN:
        return False
    path = parsed.path.lower()
    return any(snippet in path for snippet in ALLOWED_PATH_SNIPPETS)


def clean_html_to_text(html: str) -> str:
    """Turn HTML into reasonably clean plain text."""
    soup = BeautifulSoup(html, "html.parser")

    # remove noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    # keep headings as markers
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        h.insert_before("\n\n" + h.get_text(strip=True).upper() + "\n")

    text = soup.get_text(separator="\n", strip=True)

    # collapse blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def sanitize_filename(name: str) -> str:
    """
    Turn a path-like string into a safe Windows filename:
    - remove `%xx` URL-encoding chunks
    - replace illegal/unwanted chars with `_`
    - limit length to avoid MAX_PATH problems
    """
    # Remove URL encoding like %d8%b9...
    name = re.sub(r"%[0-9a-fA-F]{2}", "_", name)

    # Replace anything not alphanumeric/underscore/hyphen with underscore
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)

    # Trim repeated underscores
    name = re.sub(r"_+", "_", name).strip("_")

    # Limit length
    return name[:150] if len(name) > 150 else name


def filename_for_url(url: str) -> str:
    """Build a safe filename from the URL path."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/") or "index"
    # include query id if exists (e.g. ?id=123)
    if parsed.query:
        path += "_" + parsed.query.replace("=", "_").replace("&", "_")
    path = path.replace("/", "_")
    safe = sanitize_filename(path)
    return safe + ".txt"


# -----------------------
# MAIN SCRAPER
# -----------------------

def scrape():
    visited = set()
    queue = deque()

    for u in START_URLS:
        queue.append(normalize_url(u))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AlQudsStudentAssistantBot/1.0 (for academic project)",
        }
    )

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(visited):03d}] Fetching {url}")

        try:
            resp = session.get(url, timeout=15)
        except Exception as e:
            print("   ! Request error:", e)
            continue

        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            print("   ! Skipping (status/content-type:", resp.status_code, resp.headers.get("Content-Type"), ")")
            continue

        text = clean_html_to_text(resp.text)
        if not text:
            print("   ! Empty text, skipping")
        else:
            fn = filename_for_url(url)
            out_path = os.path.join(DATA_DIR, fn)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url}\n\n")
                    f.write(text)
                print("   ✓ Saved to", out_path)
            except OSError as e:
                # In case Windows still complains about path length etc.
                print("   ! Could not save file:", e)

        # Discover more links on this page
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(url, a["href"])
            href = normalize_url(href)
            if href not in visited and should_follow(href):
                queue.append(href)

        time.sleep(REQUEST_DELAY)

    print("\nDone. Visited", len(visited), "pages.")


if __name__ == "__main__":
    scrape()
