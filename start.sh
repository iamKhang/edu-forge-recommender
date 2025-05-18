#!/bin/bash
set -e

# Chạy migrations
echo "Running migrations..."
python /app/manage.py migrate

# Khởi động server
echo "Starting server..."
python /app/manage.py runserver 0.0.0.0:8000 