#!/bin/bash
set -e

# Tạo thư mục data nếu chưa tồn tại
mkdir -p /app/data

# Kiểm tra và thiết lập database
if [ -f /app/db.sqlite3 ] && [ ! -f /app/data/db.sqlite3 ]; then
  # Di chuyển database hiện tại vào thư mục data
  cp /app/db.sqlite3 /app/data/db.sqlite3
  rm /app/db.sqlite3
  ln -sf /app/data/db.sqlite3 /app/db.sqlite3
elif [ ! -f /app/data/db.sqlite3 ]; then
  # Tạo database mới trong thư mục data
  touch /app/data/db.sqlite3
  ln -sf /app/data/db.sqlite3 /app/db.sqlite3
elif [ ! -L /app/db.sqlite3 ]; then
  # Tạo symlink nếu chưa tồn tại
  ln -sf /app/data/db.sqlite3 /app/db.sqlite3
fi

# Chạy migrations trước để Django tạo bảng đúng cách
echo "Running migrations..."
python /app/manage.py migrate || {
  # Nếu migrations thất bại vì bảng đã tồn tại, chạy fake initial migrations
  echo "Migration failed, trying to fake initial migrations..."
  python /app/manage.py migrate --fake-initial
}

# Kiểm tra bảng recommender_user có tồn tại không
echo "Verifying tables..."
sqlite3 /app/db.sqlite3 "SELECT name FROM sqlite_master WHERE type='table';"

# Khởi động server với Gunicorn thay vì development server
echo "Starting server with Gunicorn..."
gunicorn edu_forge_recommender.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120 