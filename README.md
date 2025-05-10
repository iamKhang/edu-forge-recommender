# Edu Forge Recommender

Hệ thống đề xuất nội dung học tập thông minh sử dụng kết hợp nhiều thuật toán recommendation.

## 1. Tổng quan về các thuật toán

Hệ thống sử dụng 3 phương pháp đề xuất chính:

### 1.1. Collaborative Filtering với Deep Learning
- Sử dụng mạng neural network để học các embedding của users và posts
- Tự động phát hiện các pattern ẩn trong dữ liệu tương tác
- Có khả năng học các mối quan hệ phi tuyến tính

### 1.2. Content-Based Filtering
- Phân tích nội dung bài viết để tìm các đặc trưng quan trọng
- Xây dựng profile người dùng dựa trên nội dung họ đã tương tác
- Đề xuất các bài viết có nội dung tương tự

### 1.3. Hybrid Filtering
- Kết hợp cả hai phương pháp trên để tận dụng ưu điểm của mỗi phương pháp
- Giảm thiểu vấn đề cold-start
- Cân bằng giữa sở thích cá nhân và xu hướng chung

## 2. Chi tiết kỹ thuật

### 2.1. Collaborative Filtering Model

```python
# Kiến trúc mạng neural network
user_input = tf.keras.layers.Input(shape=(1,))
user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)(user_input)
user_vec = tf.keras.layers.Flatten()(user_embedding)

post_input = tf.keras.layers.Input(shape=(1,))
post_embedding = tf.keras.layers.Embedding(num_posts, embedding_dim)(post_input)
post_vec = tf.keras.layers.Flatten()(post_embedding)

concat = tf.keras.layers.Concatenate()([user_vec, post_vec])
dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
```

#### Đặc điểm:
- Embedding dimension: 32
- Hidden layers: 2 (64 và 32 neurons)
- Activation: ReLU cho hidden layers, Sigmoid cho output
- Loss function: Binary Cross Entropy
- Optimizer: Adam

### 2.2. Content-Based Filtering

#### Xử lý nội dung:
- Sử dụng TF-IDF để vector hóa nội dung
- Trích xuất 1000 features quan trọng nhất
- Tính toán similarity sử dụng cosine similarity

#### Công thức tính similarity:
```
similarity = dot(user_profile, post_vector) / (norm(user_profile) * norm(post_vector))
```

### 2.3. Hybrid Filtering

#### Công thức kết hợp:
```
final_score = (collab_score * 0.6) + (content_score * 0.4)
```

## 3. Quy trình xử lý dữ liệu

### 3.1. Thu thập dữ liệu
- Lấy dữ liệu từ API: `http://localhost:8080/api/posts/training-data/all`
- Bao gồm:
  - Thông tin bài viết (id, tags, content)
  - Tương tác người dùng (views, likes)

### 3.2. Tiền xử lý
1. **Xử lý tags:**
   - Chuyển đổi tags thành vector nhị phân
   - Sử dụng MultiLabelBinarizer

2. **Xử lý nội dung:**
   - Vector hóa nội dung với TfidfVectorizer
   - Lưu trữ content embeddings

3. **Xử lý tương tác:**
   - Tạo ma trận tương tác user-post
   - Ghi nhận views và likes

### 3.3. Training
- Epochs: 10
- Batch size: 64
- Validation split: 20%
- Positive samples: Các cặp (user, post) có tương tác
- Negative samples: Chọn ngẫu nhiên các bài viết chưa tương tác

## 4. API Endpoints

### 4.1. Train Model
```
POST /api/v1/interactions/retrain_model/
```
Response:
```json
{
    "message": "Model retrained successfully",
    "stats": {
        "num_users": 100,
        "num_posts": 50,
        "model_saved": true,
        "embeddings_saved": true
    }
}
```

### 4.2. Train Model với Toàn bộ Dữ liệu
```
POST /api/v1/interactions/train_all_data/
```
Response:
```json
{
    "message": "Model trained successfully on all data",
    "stats": {
        "num_users": 100,
        "num_posts": 50,
        "num_interactions": 1000,
        "model_saved": true,
        "embeddings_saved": true,
        "training_data_size": 150
    }
}
```

### 4.3. Lấy Recommendations
```
GET /api/v1/interactions/get_recommendations/?user_id=<id>
```
Response:
```json
{
    "user_id": "user_id",
    "embedding": [...],
    "similar_users": [
        {
            "user_id": "similar_user_id",
            "similarity": 0.75
        }
    ],
    "collaborative_recommendations": ["post_id1", "post_id2", ...],
    "content_based_recommendations": ["post_id1", "post_id2", ...],
    "hybrid_recommendations": ["post_id1", "post_id2", ...]
}
```

## 5. Cài đặt và Chạy

1. Tạo môi trường ảo:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Chạy migrations:
```bash
python manage.py migrate
```

4. Khởi động server:
```bash
python manage.py runserver
```

## 6. Lưu ý quan trọng

1. **Tính ngẫu nhiên:**
   - Model sử dụng tính ngẫu nhiên trong training
   - Mỗi lần train cho kết quả khác nhau
   - Giúp học được nhiều pattern khác nhau

2. **Hiệu suất:**
   - Lần đầu gọi API sẽ mất thời gian để train
   - Các lần sau sẽ nhanh hơn
   - Đảm bảo API training data có thể truy cập

3. **Chất lượng recommendations:**
   - Phụ thuộc vào chất lượng dữ liệu training
   - Càng nhiều tương tác, càng chính xác
   - Có thể điều chỉnh hyperparameters

4. **Ưu điểm của Hybrid Filtering:**
   - Kết hợp ưu điểm của cả 2 phương pháp
   - Giảm cold-start problem
   - Đa dạng hóa recommendations
   - Cân bằng sở thích cá nhân và xu hướng chung