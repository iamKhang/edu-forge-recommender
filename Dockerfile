FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Make start script executable
RUN chmod +x /app/start.sh

# Run server
CMD ["/app/start.sh"] 