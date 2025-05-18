FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create model directory with proper permissions
RUN mkdir -p /app/recommender/model && chmod -R 777 /app/recommender/model

# Create static files directory
RUN mkdir -p /app/static
RUN python manage.py collectstatic --noinput || echo "No static files to collect"

# Expose port
EXPOSE 8000

# Run server with increased timeout and worker settings
CMD ["gunicorn", "edu_forge_recommender.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "1200", "--keep-alive", "65", "--log-level", "debug"]