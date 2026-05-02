FROM python:3.11-slim

WORKDIR /app

ARG INSTALL_CHROMA=false

COPY requirements.txt requirements-chroma.txt ./
RUN if [ "$INSTALL_CHROMA" = "true" ]; then \
      pip install --no-cache-dir -r requirements-chroma.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app ./app
RUN mkdir -p /app/data

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
