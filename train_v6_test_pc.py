import cv2
import dlib
import numpy as np
import tensorflow as tf
import time

# 1. 모델 및 탐지기 로드
model = tf.keras.models.load_model('drowsy_multimodal_v6_retry.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 2. 변수 설정
closed_stack = 0
yawn_count = 0
is_yawning_now = False
last_yawn_time = 0
last_cnn_pred = 1.0
alert_active = False

# 임계값 및 타이머
EAR_THRESHOLD, CNN_THRESHOLD, MAR_THRESHOLD = 0.18, 0.45, 0.55
SLEEP_ALERT_ON, SLEEP_ALERT_OFF = 150, 0
RESET_DURATION, BLINK_TOLERANCE = 10.0, 0.5

# --- 🎯 [구역 잠금 고도화] ---
target_locked = False
driver_seat_center = None  
DRIVING_ZONE_RADIUS = 80    # 선우님 요청 반영: 범위를 더 좁게 설정 (150 -> 80)
HEAD_DOWN_TRIGGER_STACK = 40 
# ----------------------------

normal_start_time = 0   
blink_buffer_start = 0
prev_time = time.time()

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

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy() 
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    current_ear, current_mar, display_normal_time = 1.0, 0.0, 0.0
    status_msg = ""
    current_target = None

    # --- 🎯 배타적 운전자 구역 필터링 ---
    if faces:
        if not target_locked:
            # 최초 인식 시 중앙의 운전자를 타겟으로 고정
            main_driver = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
            driver_seat_center = np.array([(main_driver.left() + main_driver.right()) / 2, 
                                          (main_driver.top() + main_driver.bottom()) / 2])
            target_locked = True
            current_target = main_driver
        else:
            # 좁은 반경(80px) 내에 있는 얼굴만 운전자로 인정
            valid_candidates = []
            for f in faces:
                face_center = np.array([(f.left() + f.right()) / 2, (f.top() + f.bottom()) / 2])
                if np.linalg.norm(face_center - driver_seat_center) < DRIVING_ZONE_RADIUS:
                    valid_candidates.append(f)
            
            if valid_candidates:
                current_target = min(valid_candidates, key=lambda f: np.linalg.norm(
                    np.array([(f.left()+f.right())/2, (f.top()+f.bottom())/2]) - driver_seat_center
                ))

    # --- 🔵 인식 성공 시 ---
    if current_target:
        # 위치 미세 업데이트
        new_center = np.array([(current_target.left() + current_target.right()) / 2, 
                               (current_target.top() + current_target.bottom()) / 2])
        driver_seat_center = 0.9 * driver_seat_center + 0.1 * new_center
        
        landmarks = predictor(gray, current_target)
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        current_ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
        m_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        current_mar = get_mar(m_pts)

        # CNN 추론
        preds = []
        for eye_pts in [left_eye, right_eye]:
            ex1, ey1, ex2, ey2 = np.min(eye_pts[:,0]), np.min(eye_pts[:,1]), np.max(eye_pts[:,0]), np.max(eye_pts[:,1])
            roi = clean_frame[max(0,ey1-10):min(h,ey2+10), max(0,ex1-10):min(w,ex2+10)]
            if roi.size > 0:
                roi_in = cv2.resize(roi, (224, 224)) / 255.0
                preds.append(model.predict(np.expand_dims(roi_in, axis=0), verbose=0)[0][0])
        if preds: last_cnn_pred = np.mean(preds)

        # 판정 로직 (10초 리셋)
        is_eye_closed = (current_ear < EAR_THRESHOLD or last_cnn_pred < CNN_THRESHOLD)
        if is_eye_closed:
            closed_stack += 3
            if blink_buffer_start == 0: blink_buffer_start = time.time()
            if time.time() - blink_buffer_start > BLINK_TOLERANCE: normal_start_time = 0
        else:
            closed_stack = max(0, closed_stack - 1)
            blink_buffer_start = 0
            if normal_start_time == 0: normal_start_time = time.time()
            display_normal_time = time.time() - normal_start_time
            if display_normal_time >= RESET_DURATION:
                closed_stack = 0
                alert_active = False

        # --- 🖼️ UI: 메인 박스 및 확대 창 ---
        box_color = (0, 0, 255) if is_eye_closed else (0, 255, 0)
        cv2.rectangle(frame, (current_target.left(), current_target.top()), (current_target.right(), current_target.bottom()), box_color, 2)
        cv2.putText(frame, "DRIVER", (current_target.left(), current_target.top()-10), 1, 1, box_color, 2)

        cv2.rectangle(frame, (w-185, 65), (w-5, 345), (45, 45, 45), -1)
        for i, pts in enumerate([left_eye, right_eye]):
            ex1, ey1, ex2, ey2 = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
            roi_z = clean_frame[max(0,ey1-15):min(h,ey2+15), max(0,ex1-15):min(w,ex2+15)].copy()
            if roi_z.size > 0:
                zoom = cv2.resize(roi_z, (110, 65))
                for pt in pts:
                    px = int((pt[0]-(ex1-15))*(110/roi_z.shape[1])); py = int((pt[1]-(ey1-15))*(65/roi_z.shape[0]))
                    cv2.circle(zoom, (px, py), 2, (0, 255, 0), -1)
                frame[75+(i*80):140+(i*80), w-165:w-55] = zoom
        
        mx1, my1, mx2, my2 = np.min(m_pts[:,0]), np.min(m_pts[:,1]), np.max(m_pts[:,0]), np.max(m_pts[:,1])
        m_roi = clean_frame[max(0,my1-15):min(h,my2+15), max(0,mx1-15):min(w,mx2+15)].copy()
        if m_roi.size > 0:
            m_zoom = cv2.resize(m_roi, (110, 65))
            for pt in m_pts[12:]:
                px = int((pt[0]-(mx1-15))*(110/m_roi.shape[1])); py = int((pt[1]-(my1-15))*(65/m_roi.shape[0]))
                cv2.circle(m_zoom, (px, py), 2, (0, 255, 255), -1)
            frame[240:305, w-165:w-55] = m_zoom

        if current_mar > MAR_THRESHOLD:
            if not is_yawning_now:
                yawn_count += 1
                is_yawning_now = True
                last_yawn_time = time.time()
        else: is_yawning_now = False

    # --- 🔴 인식 실패 시 (고개 숙임 등) ---
    else:
        if closed_stack >= HEAD_DOWN_TRIGGER_STACK:
            closed_stack += 2
            status_msg = "HEAD DOWN DETECTED"
        else:
            status_msg = "SEARCHING IN ZONE..."

    # --- 📊 상단 대시보드 ---
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    dash_info = f"FPS:{fps:.1f} | EAR:{current_ear:.2f} | MAR:{current_mar:.2f} | CNN:{last_cnn_pred:.2f}"
    stack_info = f"S:{closed_stack} | Y:{yawn_count} | OK:{display_normal_time:.1f}s"
    cv2.putText(frame, dash_info, (15, 22), 1, 0.9, (255, 255, 255), 1)
    cv2.putText(frame, stack_info, (15, 45), 1, 0.9, (0, 255, 255), 1)

    if closed_stack >= SLEEP_ALERT_ON: alert_active = True
    if closed_stack <= SLEEP_ALERT_OFF: alert_active = False

    if alert_active or status_msg == "HEAD DOWN DETECTED":
        cv2.rectangle(frame, (0, h-70), (w, h), (0, 0, 180), -1)
        msg = "!!! EMERGENCY: WAKE UP !!!" if alert_active else "!!! WARNING: HEAD DOWN !!!"
        cv2.putText(frame, msg, (w//2-220, h-25), 1, 1.8, (255, 255, 255), 3)

    # 구역 시각화 (확인용)
    if target_locked:
        cv2.circle(frame, (int(driver_seat_center[0]), int(driver_seat_center[1])), DRIVING_ZONE_RADIUS, (255, 255, 255), 1)

    cv2.imshow("v6 Super Locked DMS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
