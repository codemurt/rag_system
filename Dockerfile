FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/chroma_db /app/logs /app/.cache/huggingface /app/uploads && \
    chmod -R 777 /app/chroma_db /app/logs /app/.cache/huggingface /app/uploads

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/chroma_db /app/logs /app/.cache/huggingface

COPY app.py .
COPY .env .

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
