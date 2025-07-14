FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    psmisc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install psutil \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "start.py"]