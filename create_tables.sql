-- Tạo bảng recommender_user nếu chưa tồn tại
CREATE TABLE IF NOT EXISTS recommender_user (
    id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    created_at TIMESTAMP
);

-- Tạo bảng recommender_post nếu chưa tồn tại
CREATE TABLE IF NOT EXISTS recommender_post (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMP
);

-- Tạo bảng recommender_userpostinteraction nếu chưa tồn tại
CREATE TABLE IF NOT EXISTS recommender_userpostinteraction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    post_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES recommender_user(id),
    FOREIGN KEY (post_id) REFERENCES recommender_post(id)
); 