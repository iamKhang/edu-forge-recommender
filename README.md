# Edu Forge Recommender

A Django REST Framework application with TensorFlow integration for course recommendations.

## Thuật toán Recommendation Engine

### 1. Tổng quan
Hệ thống sử dụng mô hình Deep Learning kết hợp với Collaborative Filtering để tạo ra các đề xuất bài viết cho người dùng. Mô hình học cách biểu diễn người dùng và bài viết trong không gian embedding, sau đó sử dụng các biểu diễn này để dự đoán mức độ phù hợp giữa người dùng và bài viết.

### 2. Kiến trúc Model

```python
# User input
user_input = tf.keras.layers.Input(shape=(1,))
user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)(user_input)
user_vec = tf.keras.layers.Flatten()(user_embedding)

# Post input
post_input = tf.keras.layers.Input(shape=(1,))
post_embedding = tf.keras.layers.Embedding(num_posts, embedding_dim)(post_input)
post_vec = tf.keras.layers.Flatten()(post_embedding)

# Merge layers
concat = tf.keras.layers.Concatenate()([user_vec, post_vec])
dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
```

### 3. Quy trình xử lý dữ liệu

#### 3.1. Thu thập dữ liệu
- Lấy dữ liệu từ API: `http://localhost:8080/api/posts/training-data/all`
- Dữ liệu bao gồm:
  - Thông tin bài viết (id, tags, content)
  - Tương tác người dùng (views, likes)

#### 3.2. Tiền xử lý
1. **Xử lý tags:**
   - Chuyển đổi tags thành vector nhị phân sử dụng MultiLabelBinarizer
   - Mỗi bài viết được biểu diễn bằng một vector 0/1

2. **Xử lý nội dung:**
   - Sử dụng TfidfVectorizer để chuyển đổi nội dung thành vector
   - Trích xuất 1000 features quan trọng nhất

3. **Xử lý tương tác:**
   - Tạo ma trận tương tác người dùng-bài viết
   - Ghi nhận các tương tác: views và likes

### 4. Quá trình Training

#### 4.1. Chuẩn bị dữ liệu training
1. **Positive samples:**
   - Các cặp (user, post) có tương tác
   - Label = 1

2. **Negative samples:**
   - Chọn ngẫu nhiên các bài viết chưa tương tác
   - Số lượng bằng với số positive samples
   - Label = 0

#### 4.2. Training model
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Metrics: Accuracy
- Epochs: 10
- Batch size: 64
- Validation split: 20%

### 5. Tạo Recommendations

#### 5.1. Tính toán similarity
- Sử dụng cosine similarity giữa user embedding và post embedding
- Công thức: 
```
similarity = dot(user_embedding, post_embedding) / (norm(user_embedding) * norm(post_embedding))
```

#### 5.2. Tìm users tương tự
- Tính similarity giữa user hiện tại với tất cả users khác
- Sắp xếp theo độ tương đồng giảm dần
- Lấy top 5 users tương tự nhất

#### 5.3. Đề xuất bài viết
- Tính similarity giữa user với tất cả bài viết
- Sắp xếp theo độ tương đồng giảm dần
- Lấy top 5 bài viết phù hợp nhất

### 6. API Endpoints

#### 6.1. Lấy recommendations cho một user
```
GET /api/v1/interactions/get_recommendations/?user_id=<id>
```
Response:
```json
{
  "user_id": "user_id",
  "embedding": [...],  // Vector 32 chiều
  "similar_users": [
    {
      "user_id": "similar_user_id",
      "similarity": 0.75
    }
  ],
  "recommended_posts": ["post_id1", "post_id2", ...]
}
```

#### 6.2. Lấy recommendations cho tất cả users
```
GET /api/v1/interactions/get_all_recommendations/
```
Response: Mảng các user profiles với recommendations

### 7. Cài đặt và Chạy

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

### 8. Lưu ý quan trọng

1. **Tính ngẫu nhiên:**
   - Model sử dụng tính ngẫu nhiên trong quá trình training
   - Mỗi lần train sẽ cho kết quả khác nhau
   - Giúp model học được nhiều pattern khác nhau

2. **Hiệu suất:**
   - Lần đầu gọi API sẽ mất thời gian để train model
   - Các lần sau sẽ nhanh hơn
   - Đảm bảo API training data có thể truy cập được

3. **Chất lượng recommendations:**
   - Phụ thuộc vào chất lượng dữ liệu training
   - Càng nhiều tương tác, recommendations càng chính xác
   - Có thể điều chỉnh các hyperparameters để cải thiện kết quả

## Setup

1. Create a virtual environment:
```