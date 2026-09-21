FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY pyproject.toml ./pyproject.toml

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "scripts.apscheduler_runner"]