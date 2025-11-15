
## 📂 Cấu trúc Thư mục
anime-recommender/
├── 📁 Dataset/
│   ├── 📄 anime.csv                # Thông tin chi tiết của anime
│   ├── 📄 rating.csv               # Bảng rating của người dùng (dữ liệu bạn dùng)
│   ├── 📄 train_rating.csv         # Tập train
│   ├── 📄 test_rating.csv          # Tập test
│   ├── 📄 preprocessed.csv         # Dữ liệu anime đã được tiền xử lý
│   └── 📄 matrix_genre.csv         # Ma trận onehot genre của từng anime
│
├── 📄 01-preprocess.ipynb               # Tiền xử lý tạo file preprocessed.csv và matrix_gerne.csv
├── 📄 02-eda.ipynb                      # EDA dữ liệu (có sử dụng matrix_genre.csv)
├── 📄 03-ground-truth.ipynb             # Tạo tập train_rating.csv và test_rating.csv
├── 📄 04-embedding_eda.ipynb            # Tiền xử lý cho Count Vector, tạo tập embedding_anime.csv
├── 📄 05-als-implicit.ipynb             # Collaborative Filtering với ALS, train và đánh giá
├── 📄 06-item-based-light.ipynb         # Item-based Filtering với Count Vector, train và đánh giá
├── 📄 07-TFIDF.ipynb                    # Item-based Filtering với TFIDF, train và đánh giá
├── 📄 08-hybrid.ipynb                   # Hybrid với ALS và Count Vector
├── 📄 09-clustering_pcs_kmeans.ipynb    # Phân tích PCA và gom cụm
├── 📄 10-metrics_eval.py                # Module đánh giá
├── 📄 .gitignore                      
└── 📄 README.md                         # File mô tả dự án

### 🏃 Các bước chạy
1.  Cần phải chạy các notebook có vai trò sinh ra các file csv mới trước, như thứ tự đã đề cập trong mô tả trên
2.  5 Notebook cuối cùng được đề cập trong cấu trúc trên có thể chạy theo thứ tự tùy thích 
3.  Lưu ý: file clustering_pcs_kmeans.ipynb được thiết kế để chạy trên google colab, vì máy local không thể chạy bước phân tích PCA.