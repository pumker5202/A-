import cv2
import dlib
import numpy as np
import keras
import time
from flask import Flask, Response
import os
import threading
from queue import Queue

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

# 1. 모델 로드
print("🔄 운전자 이탈 시 중앙 리셋 모드 로딩 중...")
try:
    model = keras.models.load_model('drowsy_multimodal_v6_retry.h5', compile=False)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    print("✅ 준비 완료!")
except Exception as e:
    print(f"❌ 로딩 실패: {e}")

# --- 🎯 핵심 변수 ---
eye_closed_start_time = 0
eye_closed_duration = 0.0
yawn_count, is_yawning_now = 0, False
yawn_alert_time = 0
last_cnn_pred = 1.0
alert_active = False
is_currently_sleeping = False 

EAR_THRESHOLD, CNN_THRESHOLD, MAR_THRESHOLD = 0.17, 0.45, 0.50
SLEEP_TIME_LIMIT = 10.0  
BLINK_TOLERANCE = 0.8  

# 운전자 추적 변수 (초기값 중앙 설정)
target_locked = False
driver_seat_center = np.array([320.0, 240.0]) # 화면 중앙(640x480 기준)
DRIVING_ZONE_RADIUS = 110 
face_lost_start_time = 0

last_frame_data = {
    'l_eye': None, 'r_eye': None, 'm_pts': None,
    'ear': 1.0, 'mar': 0.0, 'box': None, 'box_color': (0, 255, 0)
}

normal_start_time, display_normal_time = 0, 0.0
output_frame = None
lock = threading.Lock()
request_queue = Queue(maxsize=1)
frame_count = 0

def get_ear(eye_points):
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c)

def get_mar(mouth_points):
    top = np.linalg.norm(mouth_points[13] - mouth_points[19])
    mid = np.linalg.norm(mouth_points[14] - mouth_points[18])
    bottom = np.linalg.norm(mouth_points[15] - mouth_points[17])
    horizontal = np.linalg.norm(mouth_points[12] - mouth_points[16])
    return (top + mid + bottom) / (3.0 * horizontal)

def cnn_worker():
    global last_cnn_pred
    while True:
        rois = request_queue.get()
        if rois is None: break
        try:
            preds = []
            for roi in rois:
                roi_in = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
                p = model.predict(np.expand_dims(roi_in, axis=0), verbose=0)[0][0]
                preds.append(p)
            if preds: last_cnn_pred = np.mean(preds)
        except: pass
        request_queue.task_done()

def process_video():
    global eye_closed_start_time, eye_closed_duration, yawn_count, is_yawning_now, yawn_alert_time
    global last_cnn_pred, alert_active, target_locked, driver_seat_center
    global normal_start_time, display_normal_time, output_frame, is_currently_sleeping, frame_count
    global face_lost_start_time, last_frame_data

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_loop_time = time.time()

    while True:
        success, frame = cap.read()
        if not success: break

        current_loop_time = time.time()
        dt = current_loop_time - last_loop_time
        last_loop_time = current_loop_time
        fps = 1 / dt if dt > 0 else 0
        frame_count += 1

        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 탐지 부하 조절
        faces = []
        if frame_count % 2 == 0:
            faces = detector(gray, 0)

        current_target = None
        if faces:
            if not target_locked:
                # [중앙 리셋 후 재시작] 가장 큰 얼굴을 새 운전자로 지정
                main_face = max(faces, key=lambda rect: rect.area())
                driver_seat_center = np.array([(main_face.left()+main_face.right())/2, (main_face.top()+main_face.bottom())/2])
                target_locked = True
                current_target = main_face
                face_lost_start_time = 0
            else:
                # 기존 원 근처 얼굴 탐색
                valid = [f for f in faces if np.linalg.norm(np.array([(f.left()+f.right())/2, (f.top()+f.bottom())/2]) - driver_seat_center) < DRIVING_ZONE_RADIUS]
                if valid:
                    current_target = min(valid, key=lambda f: np.linalg.norm(np.array([(f.left()+f.right())/2, (f.top()+f.bottom())/2]) - driver_seat_center))
                    new_c = np.array([(current_target.left()+current_target.right())/2, (current_target.top()+current_target.bottom())/2])
                    driver_seat_center = 0.8 * driver_seat_center + 0.2 * new_c
                    face_lost_start_time = 0

        # --- 판정 및 데이터 유지 ---
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

            is_eye_closed_instant = (ear < EAR_THRESHOLD or last_cnn_pred < CNN_THRESHOLD)
            if is_eye_closed_instant:
                if eye_closed_start_time == 0: eye_closed_start_time = current_loop_time
                eye_closed_duration = current_loop_time - eye_closed_start_time
                if eye_closed_duration > BLINK_TOLERANCE:
                    is_currently_sleeping, box_color = True, (0, 0, 255)
                    normal_start_time, display_normal_time = 0, 0.0
            else:
                is_currently_sleeping, box_color = False, (0, 255, 0)
                eye_closed_start_time, eye_closed_duration = 0, 0.0
                if normal_start_time == 0: normal_start_time = current_loop_time
                display_normal_time = current_loop_time - normal_start_time
                if display_normal_time >= 10.0: alert_active = False

            if mar > MAR_THRESHOLD:
                if not is_yawning_now: yawn_count += 1; is_yawning_now = True; yawn_alert_time = current_loop_time
            else: is_yawning_now = False

            last_frame_data.update({
                'l_eye': l_eye, 'r_eye': r_eye, 'm_pts': m_pts,
                'ear': ear, 'mar': mar, 'box_color': box_color,
                'box': (current_target.left(), current_target.top(), current_target.right(), current_target.bottom())
            })

            if request_queue.empty():
                rois = []
                for pts in [l_eye, r_eye]:
                    ex1, ey1, ex2, ey2 = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
                    roi = clean_frame[max(0,ey1-7):min(h,ey2+7), max(0,ex1-7):min(w,ex2+7)]
                    if roi.size > 0: rois.append(roi)
                if rois: request_queue.put(rois)

        elif target_locked and face_lost_start_time == 0:
            face_lost_start_time = current_loop_time

        # [유지 로직] 0.6초 이내 실종 시 데이터 보존
        if not current_target and face_lost_start_time != 0 and (current_loop_time - face_lost_start_time < 0.6):
            if last_frame_data['l_eye'] is not None:
                should_draw_ui = True
                l_eye, r_eye, m_pts = last_frame_data['l_eye'], last_frame_data['r_eye'], last_frame_data['m_pts']
                box_color = last_frame_data['box_color']
                if is_currently_sleeping: eye_closed_duration = current_loop_time - eye_closed_start_time

        if not should_draw_ui:
            if is_currently_sleeping:
                eye_closed_duration = current_loop_time - eye_closed_start_time
                time.sleep(0.01) 
            else:
                eye_closed_duration = 0.0
                normal_start_time, display_normal_time = 0, 0.0
                # [선우님 요청] 3초 이상 이탈 시 원 위치 중앙 리셋
                if face_lost_start_time != 0 and (current_loop_time - face_lost_start_time > 3.0):
                    target_locked = False
                    driver_seat_center = np.array([320.0, 240.0]) # 중앙 리셋
                    face_lost_start_time = 0

        # --- UI 드로잉 ---
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        dash = f"FPS:{fps:.1f} | EAR:{last_frame_data['ear']:.2f} | MAR:{last_frame_data['mar']:.2f} | CNN:{last_cnn_pred:.2f}"
        info = f"CLOSE:{eye_closed_duration:.1f}s | YAWN:{yawn_count} | OK:{display_normal_time:.1f}s"
        cv2.putText(frame, dash, (15, 25), 1, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info, (15, 50), 1, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

        if should_draw_ui:
            bx = last_frame_data['box']
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), box_color, 2)
            cv2.rectangle(frame, (w-185, 65), (w-5, 345), (45, 45, 45), -1)
            for i, pts in enumerate([l_eye, r_eye, m_pts]):
                ex1, ey1, ex2, ey2 = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
                roi_z = clean_frame[max(0,ey1-15):min(h,ey2+15), max(0,ex1-15):min(w,ex2+15)].copy()
                if roi_z.size > 0:
                    zoom = cv2.resize(roi_z, (110, 65), interpolation=cv2.INTER_CUBIC)
                    p_col = (0, 255, 0) if i < 2 else (0, 255, 255)
                    for pt in (pts if i < 2 else pts[12:]):
                        px, py = int((pt[0]-(ex1-15))*(110/roi_z.shape[1])), int((pt[1]-(ey1-15))*(65/roi_z.shape[0]))
                        cv2.circle(zoom, (px, py), 2, p_col, -1)
                    frame[75+(i*82):140+(i*82), w-165:w-55] = zoom

        if eye_closed_duration >= SLEEP_TIME_LIMIT: alert_active = True
        if alert_active:
            cv2.rectangle(frame, (0, h-70), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "!! EMERGENCY: SLEEP !!", (w//2-180, h-25), 1, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        elif not should_draw_ui and not is_currently_sleeping:
            cv2.rectangle(frame, (0, h-70), (w, h), (0, 180, 0), -1)
            cv2.putText(frame, "!! DRIVER LEAVE !!", (w//2-140, h-25), 1, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        
        if yawn_count >= 3 and not alert_active and (current_loop_time - yawn_alert_time < 3.0):
            cv2.rectangle(frame, (0, h-70), (w, h), (0, 200, 255), -1)
            cv2.putText(frame, f"REST ADVISED: {yawn_count} YAWNS", (w//2-200, h-25), 1, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        # [복구 및 유지] 운전자 고정 원 그리기
        cv2.circle(frame, (int(driver_seat_center[0]), int(driver_seat_center[1])), DRIVING_ZONE_RADIUS, (255, 255, 255), 1)

        with lock: output_frame = frame.copy()

def generate():
    while True:
        with lock:
            if output_frame is None: continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            f = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index(): 
    return "<html><body style='background:#000; margin:0; overflow:hidden; display:flex; justify-content:center; align-items:center; width:100vw; height:100vh;'><img src='/video_feed' style='width:100%; height:100%; object-fit:contain;'></body></html>"

if __name__ == '__main__':
    threading.Thread(target=cnn_worker, daemon=True).start()
    threading.Thread(target=process_video, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
