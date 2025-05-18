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

# Chạy SQL trực tiếp để đảm bảo bảng tồn tại
echo "Creating tables if not exists..."
sqlite3 /app/db.sqlite3 < /app/create_tables.sql

# Chạy migrations
echo "Running migrations..."
python /app/manage.py migrate

# Kiểm tra bảng recommender_user có tồn tại không
echo "Verifying tables..."
sqlite3 /app/db.sqlite3 "SELECT name FROM sqlite_master WHERE type='table';"

# Khởi động server
echo "Starting server..."
python /app/manage.py runserver 0.0.0.0:8000 