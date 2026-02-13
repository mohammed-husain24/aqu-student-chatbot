# Al-Quds University Student Assistant Chatbot

A simple AI-powered chatbot built to help students access information about Al-Quds University.

The project combines web scraping, local document storage, and a Flask backend to answer questions related to admissions, faculties, housing, scholarships, and more.

---

## What this project does

- Scrapes university website content
- Stores the content locally (Arabic + English)
- Loads text chunks for retrieval
- Uses an LLM (optional) to generate the final answer
- Provides a small web-based chat interface

---

## Project structure

- `app.py` – main Flask app  
- `scrape_alquds.py` – scraping script  
- `data/` – stored content  
- `templates/` – HTML template  
- `static/` – CSS/JS assets  

---

## Requirements

- Python 3.10+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup (API keys)

If you want to enable the optional Groq / LLM step, create a file named `code.env` (do **not** commit it) and add:

```
GROQ_API_KEY=your_key_here
```

Then run the app.

---

## Run

```bash
python app.py
```

Open in your browser:

- `http://127.0.0.1:5000`

---

## Notes

- The embedding model downloads automatically on first run.
- API keys should **never** be committed (use `code.env` / environment variables).
