---
title: Dem So Nguoi Trong Khong Gian Cong Cong
emoji: 🧑‍🤝‍🧑
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
python_version: "3.10"
pinned: false
---

# Đếm số người trong không gian công cộng

Demo sử dụng mô hình YOLOv11m (train trên CrowdHuman) để phát hiện và đếm số người
trong ảnh hoặc video.

## Cách chạy local
```bash
pip install -r requirements.txt
python app.py
```

Lưu ý: cần đặt file `best.pt` (model đã train) cùng thư mục với `app.py`.
