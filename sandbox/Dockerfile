# Filename: Dockerfile
FROM python:3.9-slim

# Make a working directory
WORKDIR /app

# Disable bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Lightweight tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Limit container memory usage to <512MB via Docker flags
CMD ["python3", "/app/run.py"]
