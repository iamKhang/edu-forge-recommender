FROM python:3.9-slim

WORKDIR /app

# Install dependencies và sqlite3
RUN apt-get update && apt-get install -y sqlite3 && apt-get clean

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Copy project files
COPY . .

# Use the data directory for database storage
RUN mkdir -p /app/data && touch /app/data/db.sqlite3 && chmod 777 /app/data/db.sqlite3
RUN chmod 777 /app/data
# Remove any existing db file and create symbolic link
RUN [ -f /app/db.sqlite3 ] && rm -f /app/db.sqlite3 || echo "No db file to remove"
RUN ln -sf /app/data/db.sqlite3 /app/db.sqlite3

# Create model directory with proper permissions
RUN mkdir -p /app/recommender/model && chmod -R 777 /app/recommender/model

# Create static files directory
RUN mkdir -p /app/static
RUN python manage.py collectstatic --noinput || echo "No static files to collect"

# Expose port
EXPOSE 8000

# Run server with migrations and increased timeout
CMD ["sh", "-c", "python manage.py migrate --fake-initial || python manage.py migrate && gunicorn edu_forge_recommender.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 600"] 