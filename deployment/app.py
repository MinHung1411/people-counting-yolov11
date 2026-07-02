import time

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLO đã train (best.pt phải nằm cùng thư mục với file này)
model = YOLO("best.pt")

# Các mức độ phân giải xử lý: giảm imgsz giúp chạy nhanh hơn trên CPU,
# đổi lại độ chính xác (đặc biệt với người ở xa/nhỏ/che khuất) giảm nhẹ
IMGSZ_MAP = {
    "Nhanh (320px)": 320,
    "Nhanh (384px)": 384,
    "Cân bằng (480px)": 480,
    "Chính xác (640px)": 640,
    "Rất chính xác (960px)": 960,
}

VIDEO_IMGSZ_CHOICES = [k for k in IMGSZ_MAP.keys() if k != "Rất chính xác (960px)"]


def detect_image(img, conf):
    """Phát hiện và đếm người trong 1 ảnh"""
    if img is None:
        return None, "Vui lòng tải ảnh lên."
    results = model(img, conf=conf, imgsz=640, verbose=False)[0]
    annotated = results.plot()
    count = len(results.boxes)
    return annotated, f"Số người phát hiện: {count}"


def detect_video(video_path, frame_skip, conf, iou, imgsz_label, min_hits, max_seconds):
    """Phát hiện + TRACK người xuyên suốt video (ByteTrack), có giới hạn
    thời gian xử lý tối đa để đảm bảo tốc độ ổn định cho mục đích demo.

    - Frame bị bỏ qua chỉ được `cap.grab()` (nhảy qua, không giải mã).
    - Video output chỉ ghi frame có chạy detect, fps đầu ra giảm theo
      frame_skip tương ứng.
    - Nếu chạm giới hạn thời gian (max_seconds), dừng sớm và xuất kết quả
      với phần đã xử lý.
    """
    if video_path is None:
        return None, "Vui lòng tải video lên."

    imgsz_val = IMGSZ_MAP.get(imgsz_label, 384)
    step = int(frame_skip) + 1
    min_hits = int(min_hits)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = max(fps / step, 1)

    out_path = "output.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    start_time = time.time()
    frame_idx = 0
    max_count = 0
    unique_ids = set()
    track_hits = {}
    first_infer = True
    truncated = False

    while True:
        if max_seconds and (time.time() - start_time) > max_seconds:
            truncated = True
            break

        if frame_idx % step == 0:
            ret, frame = cap.read()
        else:
            ret = cap.grab()
            frame = None

        if not ret:
            break

        if frame_idx % step == 0:
            results = model.track(
                frame,
                persist=not first_infer,
                tracker="bytetrack.yaml",
                conf=conf,
                iou=iou,
                imgsz=imgsz_val,
                verbose=False,
            )[0]
            first_infer = False

            annotated = frame.copy()
            confirmed_ids_this_frame = []

            if results.boxes.id is not None:
                for (x1, y1, x2, y2), c, tid in zip(
                    results.boxes.xyxy.tolist(),
                    results.boxes.conf.tolist(),
                    results.boxes.id.tolist(),
                ):
                    tid = int(tid)
                    track_hits[tid] = track_hits.get(tid, 0) + 1

                    if track_hits[tid] >= min_hits:
                        confirmed_ids_this_frame.append(tid)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            annotated,
                            f"id:{tid} person {c:.2f}",
                            (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

            count = len(confirmed_ids_this_frame)
            max_count = max(max_count, count)
            unique_ids.update(confirmed_ids_this_frame)

            cv2.putText(
                annotated,
                f"Nguoi hien tai: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out.write(annotated)

        frame_idx += 1

    cap.release()
    out.release()

    total_unique = len(unique_ids) if unique_ids else max_count
    elapsed = time.time() - start_time

    lines = [
        f"Số người tối đa cùng lúc trong khung hình: {max_count}",
        f"Tổng số người ước tính xuất hiện trong video: {total_unique}",
        f"Thời gian xử lý: {elapsed:.1f}s",
    ]
    if truncated and total_frames:
        progress_pct = min(100, round(frame_idx / total_frames * 100))
        lines.append(
            f"Đã xử lý {progress_pct}% video (dừng sớm do đạt giới hạn thời gian xử lý)."
        )
    summary = "\n".join(lines)
    return out_path, summary


def detect_webcam(frame, conf, iou, imgsz_label):
    if frame is None:
        return None, ""
    imgsz_val = IMGSZ_MAP.get(imgsz_label, 384)
    results = model(frame, conf=conf, iou=iou, imgsz=imgsz_val, verbose=False)[0]
    annotated = results.plot()
    count = len(results.boxes)
    return annotated, f"Số người hiện tại: {count}"


with gr.Blocks(title="Đếm số người trong không gian công cộng") as demo:
    gr.Markdown(
        """
        # 🧑‍🤝‍🧑 Đếm số người trong không gian công cộng
        Phát hiện và đếm số người trong ảnh, video, hoặc camera trực tiếp bằng mô hình
        **YOLOv11m** (train trên bộ dữ liệu CrowdHuman).
        """
    )

    with gr.Tab("📷 Ảnh"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Ảnh đầu vào")
            img_out = gr.Image(label="Kết quả")
        img_conf = gr.Slider(
            0.1, 0.9, value=0.45, step=0.05,
            label="Ngưỡng tin cậy (conf) - tăng lên nếu bị nhận nhầm bóng/vật thể thành người",
        )
        count_out = gr.Textbox(label="Kết quả đếm")
        img_btn = gr.Button("🔍 Phát hiện", variant="primary")
        img_btn.click(detect_image, inputs=[img_in, img_conf], outputs=[img_out, count_out])

    with gr.Tab("🎬 Video"):
        with gr.Row():
            vid_in = gr.Video(label="Video đầu vào")
            vid_out = gr.Video(label="Video kết quả")
        with gr.Row():
            frame_skip = gr.Slider(0, 15, value=5, step=1, label="Bỏ qua khung hình")
            vid_conf = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Ngưỡng tin cậy (Confidence)")
        with gr.Row():
            vid_iou = gr.Slider(0.1, 0.9, value=0.6, step=0.05, label="Ngưỡng IoU")
            vid_min_hits = gr.Slider(1, 10, value=5, step=1, label="Số khung hình xác nhận")
        with gr.Row():
            vid_imgsz = gr.Dropdown(
                choices=VIDEO_IMGSZ_CHOICES, value="Nhanh (384px)",
                label="Độ phân giải xử lý",
            )
            vid_max_seconds = gr.Slider(
                30, 200, value=120, step=10,
                label="Giới hạn thời gian xử lý (giây)",
            )
        vid_count_out = gr.Textbox(label="Kết quả đếm")
        vid_btn = gr.Button("🔍 Xử lý video", variant="primary")
        vid_btn.click(
            detect_video,
            inputs=[vid_in, frame_skip, vid_conf, vid_iou, vid_imgsz, vid_min_hits, vid_max_seconds],
            outputs=[vid_out, vid_count_out],
        )

    with gr.Tab("🎥 Camera trực tiếp"):
        with gr.Row():
            cam_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
            cam_out = gr.Image(label="Kết quả trực tiếp")
        with gr.Row():
            cam_conf = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="Ngưỡng tin cậy (Confidence)")
            cam_iou = gr.Slider(0.1, 0.9, value=0.6, step=0.05, label="Ngưỡng IoU")
            cam_imgsz = gr.Dropdown(
                choices=VIDEO_IMGSZ_CHOICES, value="Nhanh (384px)",
                label="Độ phân giải xử lý",
            )
        cam_count = gr.Textbox(label="Số người hiện tại")
        cam_in.stream(
            detect_webcam,
            inputs=[cam_in, cam_conf, cam_iou, cam_imgsz],
            outputs=[cam_out, cam_count],
            time_limit=60,
            stream_every=0.5,
            concurrency_limit=1,
        )

    gr.Markdown(
        "---\n*Đồ án cuối kì môn Thị giác máy tính và Ứng dụng - Nhóm 18*"
    )

demo.launch()
