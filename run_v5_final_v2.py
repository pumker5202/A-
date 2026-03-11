import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# 1. 모델 및 탐지기 로드
print("🚀 [DMS v5] 튜닝된 시스템을 시작합니다...")
model = load_model('drowsy_multimodal_v5.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_mar(mouth_points):
    """입 벌림 수치(MAR) 계산 함수"""
    top = np.linalg.norm(mouth_points[13] - mouth_points[19])
    mid = np.linalg.norm(mouth_points[14] - mouth_points[18])
    bottom = np.linalg.norm(mouth_points[15] - mouth_points[17])
    horizontal = np.linalg.norm(mouth_points[12] - mouth_points[16])
    return (top + mid + bottom) / (3.0 * horizontal)

cap = cv2.VideoCapture(0)
closed_stack = 0

# --- ⚙️ 선우님의 커스텀 임계값 설정 ---
MAR_THRESHOLD = 0.6          # 하품 판정 수치
YAWN_ALERT_STACK = 20        # 하품 경고 스택 (약 0.7초)
SLEEP_ALERT_STACK = 60       # 수면 경고 스택 (약 2.0초)
# -----------------------------------

print("✅ 시스템 가동 중... 'q'를 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # 1. dlib 포인트 시각화 (디버깅용)
        for n in range(36, 48): # 눈: 초록색
            cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 0), -1)
        for n in range(48, 68): # 입: 노란색
            cv2.circle(frame, (landmarks.part(n).x, landmarks.part(n).y), 2, (0, 255, 255), -1)

        # 2. MAR 및 눈 상태 분석
        m_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        mar_val = get_mar(m_pts)

        eye_indices = [range(36, 42), range(42, 48)]
        preds = []
        for indices in eye_indices:
            pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in indices])
            x_min, y_min = np.min(pts, axis=0); x_max, y_max = np.max(pts, axis=0)
            roi = frame[max(0,y_min-5):min(frame.shape[0],y_max+5), 
                        max(0,x_min-5):min(frame.shape[1],x_max+5)]
            
            if roi.size > 0:
                roi_input = cv2.resize(roi, (64, 64)) / 255.0
                roi_input = np.expand_dims(roi_input, axis=0)
                prediction = model.predict(
                    {'img_input': roi_input, 'mar_input': np.array([[mar_val]])}, 
                    verbose=0
                )[0][0]
                preds.append(prediction)

        final_pred = np.mean(preds) if preds else 1.0

        # --- 🚨 3. 상태 판정 및 스택 로직 ---
        if mar_val > MAR_THRESHOLD:
            # 하품 감지 (선우님 설정: 20스택 기준)
            status = "DROWSY (Yawning)"
            color = (255, 165, 0) # 주황색
            closed_stack += 1
        elif final_pred < 0.5:
            # 수면 감지 (선우님 설정: 60스택 기준)
            status = "SLEEPING !!"
            color = (0, 0, 255) # 빨간색
            closed_stack += 1
        else:
            # 정상 상태: 스택을 빠르게 감소시켜 오작동 방지
            status = "NORMAL"
            color = (0, 255, 0) # 초록색
            closed_stack = max(0, closed_stack - 2)

        # --- 🖥️ 4. UI 대시보드 및 알람 ---
        cv2.rectangle(frame, (0, 0), (400, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {status}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"MAR: {mar_val:.2f} | Eye: {final_pred:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Stack: {closed_stack} (Y:20, S:60)", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # [이중 경고 시스템]
        if closed_stack >= SLEEP_ALERT_STACK:
            cv2.putText(frame, "!!! PULL OVER !!!", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
        elif closed_stack >= YAWN_ALERT_STACK:
            cv2.putText(frame, "!!! TAKE A BREAK !!!", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    cv2.imshow("v5 Multimodal Pro - Tuned", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
