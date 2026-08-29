FROM python:3.11-slim

# system libs needed by matplotlib / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt pytest

# copy the rest of the repository
COPY . .

# default: run the smoke tests; override to run experiments, e.g.
#   docker run --rm IMAGE python run_all.py
CMD ["pytest", "-q", "tests/"]
