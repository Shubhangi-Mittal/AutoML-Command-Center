FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt supervisor

COPY . .

ENV PORT=8000
ENV APP_DATA_DIR=/tmp/app_data
ENV UPLOAD_DIR=/tmp/uploads
ENV MODEL_DIR=/tmp/models

RUN mkdir -p /tmp/app_data /tmp/uploads /tmp/models

COPY supervisord.conf /etc/supervisord.conf

EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
