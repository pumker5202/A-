import cv2
import dlib
import numpy as np
import tensorflow as tf
from flask import Flask, Response
import time

app = Flask(__name__)

# 1. 모델 및 탐지기 로드
model = tf.keras.models.load_model('drowsy_multimodal_v5.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 2. 전역 변수
closed_stack = 0
yawn_count = 0
is_yawning_now = False
last_yawn_time = 0
last_pred = 1.0
frame_count = 0
prev_target_id = None

# --- ⚙️ 설정값 (안경/작은 눈 및 범용 인식 최적화) ---
EAR_THRESHOLD = 0.18    
CNN_THRESHOLD = 0.35    
MAR_THRESHOLD = 0.50
YAWN_ALERT_LIMIT = 3
YAWN_MSG_DURATION = 5
SLEEP_LOW_STACK = 60
SLEEP_CRITICAL_STACK = 120
EYE_MARGIN = 10         

CNN_PROCESS_INTERVAL = 8
DETECTION_INTERVAL = 15
# ----------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

def select_main_driver(faces, w, h):
    if not faces: return None, None
    center_x = w / 2
    best_face = min(faces, key=lambda f: abs((f.left()+f.right())/2 - center_x) + (10000 / ((f.right()-f.left())**2 + 1)))
    target_id = f"{best_face.left()//40}_{best_face.top()//40}"
    return best_face, target_id

def generate_frames():
    global closed_stack, yawn_count, is_yawning_now, last_yawn_time, last_pred, frame_count, prev_target_id
    
    target_face = None

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        # 박스 그리기 전 깨끗한 원본 복사 (Zoom용)
        clean_frame = frame.copy()
        
        h, w, _ = frame.shape
        alert_msg = ""
        current_ear = 1.0
        current_mar = 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % DETECTION_INTERVAL == 1:
            all_faces = detector(gray, 0)
            target_face, current_id = select_main_driver(all_faces, w, h)
            # 타겟 바뀜 감지 시 스택 초기화
            if prev_target_id != current_id:
                closed_stack = 0
                prev_target_id = current_id

        if target_face:
            landmarks = predictor(gray, target_face)
            
            # 좌표 추출
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            current_ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
            m_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
            current_mar = get_mar(m_pts)

            # --- 🔍 [Zoom 창] 박스 없는 깨끗한 화면에서 포인트 시각화 ---
            cv2.rectangle(frame, (460, 0), (640, 240), (20, 20, 20), -1)
            
            for i, pts in enumerate([left_eye, right_eye]):
                ex1, ey1 = np.min(pts, axis=0); ex2, ey2 = np.max(pts, axis=0)
                roi = clean_frame[max(0,ey1-10):min(h,ey2+10), max(0,ex1-10):min(w,ex2+10)].copy()
                if roi.size > 0:
                    zoom = cv2.resize(roi, (80, 50))
                    for pt in pts:
                        px = int((pt[0]-(ex1-10))*(80/roi.shape[1]))
                        py = int((pt[1]-(ey1-10))*(50/roi.shape[0]))
                        cv2.circle(zoom, (px, py), 1, (0, 255, 0), -1)
                    frame[10+(i*65):10+(i*65)+50, 540:620] = zoom
            
            mx1, my1 = np.min(m_pts, axis=0); mx2, my2 = np.max(m_pts, axis=0)
            m_roi = clean_frame[max(0,my1-10):min(h,my2+10), max(0,mx1-10):min(w,mx2+10)].copy()
            if m_roi.size > 0:
                m_zoom = cv2.resize(m_roi, (100, 60))
                for pt in m_pts[12:]: 
                    px = int((pt[0]-(mx1-10))*(100/m_roi.shape[1]))
                    py = int((pt[1]-(my1-10))*(60/m_roi.shape[0]))
                    cv2.circle(m_zoom, (px, py), 1, (0, 255, 255), -1)
                frame[150:210, 520:620] = m_zoom

            # --- 🟢 [메인 웹캠] 박스 시각화 (확대창 완성 후 작업) ---
            for eye_pts in [left_eye, right_eye]:
                ex1, ey1 = np.min(eye_pts, axis=0); ex2, ey2 = np.max(eye_pts, axis=0)
                cv2.rectangle(frame, (ex1-5, ey1-5), (ex2+5, ey2+5), (0, 255, 0), 1)
            cv2.rectangle(frame, (mx1-5, my1-5), (mx2+5, my2+5), (0, 255, 255), 1)

            # [CNN 분석]
            if frame_count % CNN_PROCESS_INTERVAL == 0:
                preds = []
                for eye_pts in [left_eye, right_eye]:
                    ex1, ey1 = np.min(eye_pts, axis=0); ex2, ey2 = np.max(eye_pts, axis=0)
                    roi_cnn = clean_frame[max(0,ey1-EYE_MARGIN):min(h,ey2+EYE_MARGIN), max(0,ex1-EYE_MARGIN):min(w,ex2+EYE_MARGIN)]
                    if roi_cnn.size > 0:
                        roi_in = cv2.resize(roi_cnn, (64, 64)) / 255.0
                        roi_in = roi_in[np.newaxis, ...]
                        preds.append(model.predict({'img_input': roi_in, 'mar_input': np.array([[0.1]])}, verbose=0)[0][0])
                if preds: last_pred = np.mean(preds)

            # [졸음 판정 및 스택]
            if current_ear < EAR_THRESHOLD or last_pred < CNN_THRESHOLD:
                closed_stack += 3
            else:
                closed_stack = max(0, closed_stack - 6)

            # 하품 로직
            if current_mar > MAR_THRESHOLD:
                if not is_yawning_now:
                    yawn_count += 1
                    is_yawning_now = True
                    if yawn_count >= YAWN_ALERT_LIMIT: last_yawn_time = time.time()
            else: is_yawning_now = False

            if closed_stack >= SLEEP_CRITICAL_STACK: alert_msg = "STOP IMMEDIATELY!!"
            elif closed_stack >= SLEEP_LOW_STACK: alert_msg = "DROWSY: REST NEEDED"
            elif yawn_count >= YAWN_ALERT_LIMIT and (time.time() - last_yawn_time < YAWN_MSG_DURATION):
                alert_msg = "YAWNING: TAKE A BREAK"

            # 운전자 박스
            cv2.rectangle(frame, (target_face.left(), target_face.top()), (target_face.right(), target_face.bottom()), (0, 255, 0), 1)

        # UI 출력
        cv2.rectangle(frame, (0, 0), (460, 45), (0, 0, 0), -1)
        info = f"E:{current_ear:.2f} C:{last_pred:.2f} M:{current_mar:.2f} S:{closed_stack} Y:{yawn_count}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        
        if alert_msg:
            cv2.rectangle(frame, (0, 380), (w, 480), (0, 0, 180), -1)
            cv2.putText(frame, alert_msg, (60, 440), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- Flask 라우팅 설정 ---
@app.route('/')
def index():
    return """
    <html>
        <head><title>DMS Monitor</title></head>
        <body style="background:#000; text-align:center; margin:0; padding:0;">
            <img src="/video_feed" style="width:90%; border:2px solid #333; margin-top:20px;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
