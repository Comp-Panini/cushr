# Hugging Face Spaces (Docker SDK) serves on port 7860 by convention.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ship the prebuilt demo database + app. The advisor needs nothing else.
COPY app.py .
COPY cushr_viz.db .
COPY static ./static

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
