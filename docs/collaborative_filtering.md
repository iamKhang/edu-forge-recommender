# Collaborative Filtering với Deep Learning

## 1. Giới thiệu

### 1.1. Collaborative Filtering là gì?
Collaborative Filtering (Lọc cộng tác) là một kỹ thuật đề xuất dựa trên hành vi của người dùng. Nó hoạt động dựa trên nguyên tắc: "Những người dùng có hành vi tương tự trong quá khứ sẽ có sở thích tương tự trong tương lai."

Ví dụ thực tế:
- Nếu bạn và người bạn của bạn đều thích xem phim hành động và phim hài
- Khi người bạn của bạn thích một bộ phim mới
- Hệ thống sẽ đề xuất bộ phim đó cho bạn

### 1.2. Tại sao cần Deep Learning?
- Deep Learning giúp tự động học các đặc trưng phức tạp từ dữ liệu
- Có thể phát hiện các mối quan hệ ẩn mà các phương pháp truyền thống không thấy được
- Xử lý được dữ liệu lớn và phức tạp

## 2. Các khái niệm cơ bản

### 2.1. Embedding là gì?
- Embedding là cách chuyển đổi dữ liệu (như ID người dùng, ID bài viết) thành các vector số
- Mỗi vector đại diện cho các đặc trưng ẩn của đối tượng
- Ví dụ: User ID "123" có thể được chuyển thành vector [0.1, 0.5, -0.3, ...]

### 2.2. Neural Network là gì?
- Là một mô hình máy học mô phỏng cách hoạt động của não người
- Gồm nhiều lớp (layers) xử lý thông tin
- Mỗi lớp học các đặc trưng khác nhau từ dữ liệu

### 2.3. Activation Function là gì?
- Là hàm kích hoạt giúp mạng neural network học các mối quan hệ phi tuyến
- ReLU: f(x) = max(0, x) - giúp mạng học các đặc trưng tích cực
- Sigmoid: f(x) = 1/(1 + e^(-x)) - chuyển đổi giá trị về khoảng [0,1]

## 3. Cách hoạt động

### 3.1. Quy trình xử lý
1. **Thu thập dữ liệu:**
   - Lấy thông tin về tương tác của người dùng (xem, thích)
   - Mỗi tương tác là một cặp (user_id, post_id)

2. **Chuyển đổi dữ liệu:**
   - Chuyển user_id và post_id thành các vector embedding
   - Mỗi vector có 32 chiều (dimension)

3. **Training model:**
   - Học các embedding cho users và posts
   - Tìm các mối quan hệ ẩn giữa users và posts

### 3.2. Kiến trúc mạng
```python
# Input layer
user_input = Input(shape=(1,))
post_input = Input(shape=(1,))

# Embedding layer
user_embedding = Embedding(num_users, 32)(user_input)
post_embedding = Embedding(num_posts, 32)(post_input)

# Processing layers
user_vec = Flatten()(user_embedding)
post_vec = Flatten()(post_embedding)

# Combine layers
concat = Concatenate()([user_vec, post_vec])
dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1, activation='sigmoid')(dense2)
```

## 4. Kết quả và Ý nghĩa

### 4.1. Đầu ra của model
- Mỗi user được biểu diễn bởi một vector 32 chiều
- Mỗi post được biểu diễn bởi một vector 32 chiều
- Các vector này chứa thông tin về sở thích và đặc trưng

### 4.2. Cách tạo đề xuất
1. Tính độ tương đồng giữa user và post:
```python
similarity = dot(user_vector, post_vector) / (norm(user_vector) * norm(post_vector))
```

2. Sắp xếp các post theo độ tương đồng
3. Chọn top N post có độ tương đồng cao nhất

### 4.3. Ví dụ kết quả
```json
{
    "user_id": "123",
    "recommendations": [
        {
            "post_id": "456",
            "similarity_score": 0.85
        },
        {
            "post_id": "789",
            "similarity_score": 0.75
        }
    ]
}
```

## 5. Ưu điểm và Hạn chế

### 5.1. Ưu điểm
- Tự động học các mối quan hệ phức tạp
- Không cần hiểu biết về nội dung
- Có thể phát hiện các sở thích ẩn

### 5.2. Hạn chế
- Cần nhiều dữ liệu tương tác
- Khó xử lý với người dùng mới (cold-start)
- Có thể bị ảnh hưởng bởi dữ liệu nhiễu

## 6. Ứng dụng thực tế

### 6.1. Trong hệ thống của chúng ta
- Đề xuất bài viết cho người dùng
- Tìm người dùng có sở thích tương tự
- Phân tích hành vi người dùng

### 6.2. Các ứng dụng khác
- Đề xuất phim (Netflix)
- Đề xuất sản phẩm (Amazon)
- Đề xuất nhạc (Spotify) 