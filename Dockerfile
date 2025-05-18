FROM python:3.9-slim

WORKDIR /app

# Install dependencies và sqlite3
RUN apt-get update && apt-get install -y sqlite3 && apt-get clean

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create static files directory
RUN mkdir -p /app/static
RUN python manage.py collectstatic --noinput || echo "No static files to collect"

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Run server with migrations
CMD ["sh", "-c", "python manage.py migrate || python manage.py migrate --fake-initial && gunicorn edu_forge_recommender.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120"] 