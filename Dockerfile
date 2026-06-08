FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    libgomp1 \
    libglib2.0-0 \
    psmisc \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install psutil \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "start.py"]