FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# The CPU-only torch index avoids pulling multi-GB CUDA wheels on Linux.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN groupadd --system app \
    && useradd --system --gid app --no-create-home app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /app/.cache

USER app

# Bake the embedding model into the image so containers don't download it on start.
# If EMBED_MODEL differs at runtime, it is downloaded (and cached) on first use.
ARG EMBED_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"

# Single worker on purpose: the daily APScheduler job starts inside the app lifespan,
# so extra workers would each run their own scheduler.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
