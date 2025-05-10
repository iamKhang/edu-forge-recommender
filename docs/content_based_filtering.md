# Content-Based Filtering

## 1. Giới thiệu

### 1.1. Content-Based Filtering là gì?
Content-Based Filtering (Lọc dựa trên nội dung) là kỹ thuật đề xuất dựa trên nội dung của sản phẩm và sở thích của người dùng. Nó hoạt động dựa trên nguyên tắc: "Nếu người dùng thích một sản phẩm, họ cũng sẽ thích các sản phẩm tương tự."

Ví dụ thực tế:
- Nếu bạn thích đọc bài viết về Python
- Hệ thống sẽ đề xuất các bài viết khác về Python
- Hoặc các bài viết về lập trình có liên quan

### 1.2. Tại sao cần Content-Based Filtering?
- Không cần dữ liệu về hành vi của nhiều người dùng
- Có thể đề xuất cho người dùng mới
- Đề xuất dựa trên nội dung thực tế

## 2. Các khái niệm cơ bản

### 2.1. TF-IDF là gì?
- TF (Term Frequency): Tần suất xuất hiện của từ trong văn bản
- IDF (Inverse Document Frequency): Độ quan trọng của từ trong toàn bộ tập dữ liệu
- TF-IDF = TF × IDF: Đánh giá tầm quan trọng của từ trong văn bản

Ví dụ:
- Từ "Python" xuất hiện nhiều trong bài viết về lập trình
- Từ "the" xuất hiện nhiều trong mọi bài viết
- TF-IDF của "Python" sẽ cao hơn "the"

### 2.2. Vector là gì?
- Là cách biểu diễn dữ liệu dưới dạng các số
- Mỗi số đại diện cho một đặc trưng
- Ví dụ: [0.5, 0.3, 0.8] là vector 3 chiều

### 2.3. Cosine Similarity là gì?
- Là cách đo độ tương đồng giữa hai vector
- Giá trị từ -1 đến 1
- 1: Hoàn toàn tương đồng
- 0: Không liên quan
- -1: Hoàn toàn ngược nhau

## 3. Cách hoạt động

### 3.1. Quy trình xử lý
1. **Phân tích nội dung:**
   - Đọc nội dung bài viết
   - Tách từ và xử lý văn bản
   - Chuyển thành vector TF-IDF

2. **Xây dựng user profile:**
   - Thu thập các bài viết người dùng đã tương tác
   - Tạo vector đại diện cho sở thích của người dùng
   - Cập nhật profile khi có tương tác mới

3. **Tạo đề xuất:**
   - So sánh user profile với nội dung bài viết
   - Tính độ tương đồng
   - Sắp xếp và chọn bài viết phù hợp nhất

### 3.2. Công thức tính toán
1. **TF-IDF cho một từ:**
```
TF-IDF = (số lần từ xuất hiện trong văn bản) × log(tổng số văn bản / số văn bản chứa từ)
```

2. **Cosine Similarity:**
```
similarity = dot(vector1, vector2) / (norm(vector1) × norm(vector2))
```

## 4. Kết quả và Ý nghĩa

### 4.1. Đầu ra của hệ thống
- Mỗi bài viết được biểu diễn bởi vector TF-IDF
- Mỗi user có một profile vector
- Độ tương đồng giữa user và bài viết

### 4.2. Ví dụ kết quả
```json
{
    "user_id": "123",
    "content_based_recommendations": [
        {
            "post_id": "456",
            "title": "Hướng dẫn Python cơ bản",
            "similarity_score": 0.92,
            "matched_topics": ["Python", "Lập trình", "Cơ bản"]
        },
        {
            "post_id": "789",
            "title": "Cấu trúc dữ liệu trong Python",
            "similarity_score": 0.85,
            "matched_topics": ["Python", "Cấu trúc dữ liệu"]
        }
    ]
}
```

## 5. Ưu điểm và Hạn chế

### 5.1. Ưu điểm
- Không cần dữ liệu từ nhiều người dùng
- Có thể đề xuất cho người dùng mới
- Đề xuất dựa trên nội dung thực tế
- Dễ hiểu và giải thích

### 5.2. Hạn chế
- Cần phân tích nội dung chi tiết
- Có thể bỏ sót các đề xuất không liên quan trực tiếp
- Khó phát hiện sở thích ẩn
- Phụ thuộc vào chất lượng phân tích nội dung

## 6. Ứng dụng thực tế

### 6.1. Trong hệ thống của chúng ta
- Đề xuất bài viết dựa trên nội dung
- Phân loại bài viết theo chủ đề
- Tìm kiếm bài viết tương tự

### 6.2. Các ứng dụng khác
- Đề xuất tin tức (Google News)
- Đề xuất video (YouTube)
- Đề xuất sách (Amazon Books)
- Tìm kiếm tài liệu (Google Scholar) 