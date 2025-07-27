# Filename: Dockerfile
FROM python:3.11-slim

# Make a working directory
WORKDIR /app

# Disable bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install lightweight tools & essentials
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git curl wget build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy a default entrypoint script (optional)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set entrypoint to keep container alive
ENTRYPOINT ["sleep", "infinity"]
