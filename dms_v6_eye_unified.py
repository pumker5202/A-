"""
DMS v6 - eye_classifier 단일 모델 (RKNN)

train_v6_test_lite.py 기반, Keras 대신 eye_classifier.rknn만 사용.
- 눈 뜸/감힘: eye_classifier (64x64, MobileNetV2)
- EAR, MAR, 얼굴 추적: 동일

필수 파일 (같은 폴더):
  - eye_classifier.rknn
  - shape_predictor_68_face_landmarks.dat

실행: python3 dms_v6_eye_unified.py [--no-web]
"""
import cv2
import dlib
import numpy as np
import time
import os
import threading
from pathlib import Path

try:
    from flask import Flask, Response
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "eye_classifier.rknn"
LANDMARK_PATH = SCRIPT_DIR / "shape_predictor_68_face_landmarks.dat"

# --- 설정 ---
EAR_THRESHOLD = 0.19
CNN_THRESHOLD = 0.6   # P(눈뜸) < 이 값이면 눈 감음
MAR_THRESHOLD = 0.50
SLEEP_TIME_LIMIT = 10.0
BLINK_TOLERANCE = 0.5
RECOVERY_DURATION = 0.5

rknn = None
detector = None
predictor = None
if HAS_FLASK:
    app = Flask(__name__)

eye_closed_start_time = 0
eye_closed_duration = 0.0
eye_open_start_time = 0
yawn_count, is_yawning_now = 0, False
yawn_alert_time = 0
last_cnn_pred = 1.0
alert_active = False
is_currently_sleeping = False

target_locked = False
driver_seat_center = np.array([320.0, 240.0])
DRIVING_ZONE_RADIUS = 110
face_lost_start_time = 0

last_frame_data = {
    "l_eye": None, "r_eye": None, "m_pts": None,
    "ear": 1.0, "mar": 0.0, "box": None, "box_color": (0, 255, 0),
}
normal_start_time, display_normal_time = 0, 0.0
output_frame = None
lock = threading.Lock()
frame_count = 0
use_imshow = True


def load_rknn():
    global rknn
    try:
        from rknnlite.api import RKNNLite
        r = RKNNLite()
        if r.load_rknn(str(MODEL_PATH)) != 0:
            raise RuntimeError("load_rknn 실패")
        if r.init_runtime() != 0:
            raise RuntimeError("init_runtime 실패")
        rknn = r
        return "rknn-lite2"
    except Exception:
        try:
            from rknn.api import RKNN
            r = RKNN()
            if r.load_rknn(str(MODEL_PATH)) != 0:
                raise RuntimeError("load_rknn 실패")
            if r.init_runtime(target="rk3588") != 0:
                raise RuntimeError("init_runtime 실패")
            rknn = r
            return "rknn-toolkit2"
        except ImportError:
            raise RuntimeError("rknnlite 또는 rknn 필요 (pip install rknn-toolkit-lite2)")


def infer_eye_open_prob(bgr_roi):
    """eye_classifier: 64x64, ImageNet norm → P(눈 뜸)"""
    if bgr_roi.size == 0:
        return 1.0
    roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)
    roi = roi.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    roi = (roi - mean) / std
    roi = np.transpose(roi, (2, 0, 1))
    roi = np.expand_dims(roi, axis=0).astype(np.float32)
    outs = rknn.inference(inputs=[roi])
    if not outs:
        return 1.0
    logits = np.asarray(outs[0]).reshape(-1).astype(np.float64)
    exp_ = np.exp(logits - np.max(logits))
    return float(exp_[1] / exp_.sum())


def get_ear(eye_points):
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c)


def get_mar(mouth_points):
    top = np.linalg.norm(mouth_points[13] - mouth_points[19])
    mid = np.linalg.norm(mouth_points[14] - mouth_points[18])
    bottom = np.linalg.norm(mouth_points[15] - mouth_points[17])
    hz = np.linalg.norm(mouth_points[12] - mouth_points[16])
    return (top + mid + bottom) / (3.0 * hz)


def process_video():
    global eye_closed_start_time, eye_closed_duration, eye_open_start_time
    global yawn_count, is_yawning_now, yawn_alert_time, last_cnn_pred
    global alert_active, target_locked, driver_seat_center
    global normal_start_time, display_normal_time, output_frame, is_currently_sleeping
    global face_lost_start_time, last_frame_data, frame_count

    cap = None
    for idx in range(4):
        for use_v4l2 in [True, False]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if use_v4l2 else cv2.VideoCapture(idx)
                if cap.isOpened() and cap.read()[0]:
                    break
            except Exception:
                pass
            if cap and cap.isOpened():
                break
        if cap and cap.isOpened():
            break
    if not cap or not cap.isOpened():
        raise RuntimeError("웹캠 열기 실패")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last_loop_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_loop_time = time.time()
        dt = current_loop_time - last_loop_time
        last_loop_time = current_loop_time
        fps = 1.0 / dt if dt > 0 else 0
        frame_count += 1

        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0) if frame_count % 2 == 0 else []
        current_target = None

        if faces:
            if not target_locked:
                main = max(faces, key=lambda r: r.area())
                driver_seat_center = np.array([(main.left() + main.right()) / 2, (main.top() + main.bottom()) / 2])
                target_locked = True
                current_target = main
                face_lost_start_time = 0
            else:
                valid = [f for f in faces if np.linalg.norm(
                    np.array([(f.left() + f.right()) / 2, (f.top() + f.bottom()) / 2]) - driver_seat_center
                ) < DRIVING_ZONE_RADIUS]
                if valid:
                    current_target = min(valid, key=lambda f: np.linalg.norm(
                        np.array([(f.left() + f.right()) / 2, (f.top() + f.bottom()) / 2]) - driver_seat_center))
                    nc = np.array([(current_target.left() + current_target.right()) / 2,
                                   (current_target.top() + current_target.bottom()) / 2])
                    driver_seat_center = 0.8 * driver_seat_center + 0.2 * nc
                    face_lost_start_time = 0

        should_draw_ui = False
        box_color = (0, 255, 0)

        if current_target:
            should_draw_ui = True
            landmarks = predictor(gray, current_target)
            l_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            r_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            m_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            ear = (get_ear(l_eye) + get_ear(r_eye)) / 2.0
            mar = get_mar(m_pts)

            preds = []
            for pts in [l_eye, r_eye]:
                ex1, ey1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                ex2, ey2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
                roi = clean_frame[max(0, ey1 - 7):min(h, ey2 + 7), max(0, ex1 - 7):min(w, ex2 + 7)]
                if roi.size > 0:
                    preds.append(infer_eye_open_prob(roi))
            if preds:
                last_cnn_pred = np.mean(preds)

            is_eye_closed = (ear < EAR_THRESHOLD) or (last_cnn_pred < CNN_THRESHOLD)
            if is_eye_closed:
                eye_open_start_time = 0
                if eye_closed_start_time == 0:
                    eye_closed_start_time = current_loop_time
                eye_closed_duration = current_loop_time - eye_closed_start_time
                if eye_closed_duration > BLINK_TOLERANCE:
                    is_currently_sleeping, box_color = True, (0, 0, 255)
                    normal_start_time, display_normal_time = 0, 0.0
            else:
                if not is_currently_sleeping:
                    eye_closed_start_time, eye_closed_duration = 0, 0.0
                else:
                    # 회복 구간(눈이 조금 떠짐)에서도 수면 시작 시간을 유지해서
                    # 얼굴 인식이 잠깐 끊겼을 때 CLOSE 카운트가 폭주하지 않게 함.
                    if eye_closed_start_time != 0:
                        eye_closed_duration = current_loop_time - eye_closed_start_time
                if is_currently_sleeping:
                    if eye_open_start_time == 0:
                        eye_open_start_time = current_loop_time
                    if (current_loop_time - eye_open_start_time) >= RECOVERY_DURATION:
                        is_currently_sleeping, box_color = False, (0, 255, 0)
                        normal_start_time = current_loop_time
                    else:
                        box_color = (0, 0, 255)
                else:
                    is_currently_sleeping, box_color = False, (0, 255, 0)
                    if normal_start_time == 0:
                        normal_start_time = current_loop_time
                    display_normal_time = current_loop_time - normal_start_time
                    if display_normal_time >= 10.0:
                        alert_active = False

            if mar > MAR_THRESHOLD:
                if not is_yawning_now and not is_currently_sleeping:
                    yawn_count += 1
                    is_yawning_now = True
                    yawn_alert_time = current_loop_time
            else:
                is_yawning_now = False

            last_frame_data.update({
                "l_eye": l_eye, "r_eye": r_eye, "m_pts": m_pts,
                "ear": ear, "mar": mar, "box_color": box_color,
                "box": (current_target.left(), current_target.top(), current_target.right(), current_target.bottom()),
            })

        elif target_locked and face_lost_start_time == 0:
            face_lost_start_time = current_loop_time

        if not current_target and face_lost_start_time != 0 and (current_loop_time - face_lost_start_time < 0.6):
            if last_frame_data["l_eye"] is not None:
                should_draw_ui = True
                l_eye = last_frame_data["l_eye"]
                r_eye = last_frame_data["r_eye"]
                m_pts = last_frame_data["m_pts"]
                box_color = last_frame_data["box_color"]

        if not should_draw_ui:
            eye_closed_start_time = 0
            eye_closed_duration = 0.0
            is_currently_sleeping = False
            normal_start_time, display_normal_time = 0, 0.0
            if face_lost_start_time != 0 and (current_loop_time - face_lost_start_time > 3.0):
                target_locked = False
                driver_seat_center = np.array([320.0, 240.0])
                face_lost_start_time = 0

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS:{fps:.1f} | EAR:{last_frame_data['ear']:.2f} | MAR:{last_frame_data['mar']:.2f} | CNN:{last_cnn_pred:.2f}",
                    (15, 25), 1, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CLOSE:{eye_closed_duration:.1f}s | YAWN:{yawn_count} | OK:{display_normal_time:.1f}s",
                    (15, 50), 1, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

        if should_draw_ui:
            bx = last_frame_data["box"]
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), box_color, 2)
            cv2.rectangle(frame, (w - 185, 65), (w - 5, 345), (45, 45, 45), -1)
            for i, pts in enumerate([last_frame_data["l_eye"], last_frame_data["r_eye"], last_frame_data["m_pts"]]):
                if pts is None:
                    continue
                ex1, ey1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                ex2, ey2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
                roi_z = clean_frame[max(0, ey1 - 15):min(h, ey2 + 15), max(0, ex1 - 15):min(w, ex2 + 15)].copy()
                if roi_z.size > 0:
                    zoom = cv2.resize(roi_z, (110, 65), interpolation=cv2.INTER_CUBIC)
                    p_col = (0, 255, 0) if i < 2 else (0, 255, 255)
                    for pt in (pts if i < 2 else pts[12:]):
                        px = int((pt[0] - (ex1 - 15)) * (110 / roi_z.shape[1]))
                        py = int((pt[1] - (ey1 - 15)) * (65 / roi_z.shape[0]))
                        cv2.circle(zoom, (px, py), 2, p_col, -1)
                    frame[75 + (i * 82):140 + (i * 82), w - 165:w - 55] = zoom

        if eye_closed_duration >= SLEEP_TIME_LIMIT:
            alert_active = True
        if alert_active:
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "!! EMERGENCY: SLEEP !!", (w // 2 - 180, h - 25), 1, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        elif not should_draw_ui and not is_currently_sleeping:
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 180, 0), -1)
            cv2.putText(frame, "!! DRIVER LEAVE !!", (w // 2 - 140, h - 25), 1, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        if yawn_count >= 3 and not alert_active and (current_loop_time - yawn_alert_time < 3.0):
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 200, 255), -1)
            cv2.putText(frame, f"REST ADVISED: {yawn_count} YAWNS", (w // 2 - 200, h - 25), 1, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.circle(frame, (int(driver_seat_center[0]), int(driver_seat_center[1])), DRIVING_ZONE_RADIUS, (255, 255, 255), 1)

        with lock:
            output_frame = frame.copy()

        if use_imshow:
            cv2.imshow("DMS v6 (eye_classifier)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    try:
        rknn.release()
    except Exception:
        pass


def generate():
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buf = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


if HAS_FLASK:
    @app.route("/video_feed")
    def video_feed():
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return """<html><body style='background:#000;margin:0;overflow:hidden;display:flex;justify-content:center;align-items:center;width:100vw;height:100vh;'>
            <img src='/video_feed' style='width:100%;height:100%;object-fit:contain;'>
        </body></html>"""


if __name__ == "__main__":
    import sys
    globals()["use_imshow"] = "--no-web" in sys.argv

    print("🔄 eye_classifier RKNN 로딩 중...")
    load_rknn()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(LANDMARK_PATH))
    print("✅ 준비 완료!")

    if HAS_FLASK and not use_imshow:
        print("[INFO] 웹 모드: http://0.0.0.0:5000")
        threading.Thread(target=process_video, daemon=True).start()
        app.run(host="0.0.0.0", port=5000, threaded=True)
    else:
        print("[INFO] 로컬 창 모드 (종료: q)")
        process_video()
