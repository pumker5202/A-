import cv2
import dlib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial import distance as dist

# 1. 수치 계산 함수 정의
def calculate_ear(eye):
    # 눈 세로 거리 / 가로 거리 비율
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    # 입술 안쪽 랜드마크(60~67)를 이용한 개구도 계산
    A = dist.euclidean(mouth[61], mouth[67])
    B = dist.euclidean(mouth[63], mouth[65])
    C = dist.euclidean(mouth[60], mouth[64])
    return (A + B) / (2.0 * C)

# 2. 임계값 설정 (선우님 맞춤형 수치)
# 사진상 평소 눈 수치가 0.209이므로, 0.14 미만일 때만 졸음으로 인식합니다.
THRESH_EAR = 0.14  
# 입을 크게 벌렸을 때(하품)의 기준값
THRESH_MAR = 0.45  

# 3. 모델 및 예측기 로드
# 경로가 선우님의 실제 환경과 맞는지 확인하세요.
model = YOLO("runs/detect/pure_eye_mouth_model/weights/best.pt")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

print("--- 졸음운전 감지 시스템 가동 중 ---")
print("종료하려면 'q'를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # YOLO 추론 (얼굴, 눈, 입 위치 파악)
    results = model(frame, conf=0.5, verbose=False)
    
    # r.plot() 결과물을 기반으로 출력 프레임 생성
    display_frame = frame.copy()

    for r in results:
        display_frame = r.plot() # YOLO 박스 그리기
        
        for box in r.boxes:
            # 0번 클래스(Face) 박스가 보일 때만 dlib 실행
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # dlib 랜드마크 추출 (얼굴 박스 영역 사용)
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                landmarks = predictor(frame, dlib_rect)
                pts = np.array([[p.x, p.y] for p in landmarks.parts()])

                # EAR(눈) 및 MAR(입) 실시간 수치 계산
                ear = (calculate_ear(pts[36:42]) + calculate_ear(pts[42:48])) / 2.0
                mar = calculate_mar(pts)

                # --- 시각화: 눈과 입 주변에 초록색 점 표시 ---
                for n in list(range(36, 48)) + list(range(60, 68)):
                    cv2.circle(display_frame, (pts[n][0], pts[n][1]), 1, (0, 255, 0), -1)

                # --- 판정 및 경고 메시지 출력 ---
                # 1. 졸음 감지 (빨간색)
                if ear < THRESH_EAR:
                    cv2.putText(display_frame, "!!! DROWSY ALERT !!!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                
                # 2. 하품 감지 (노란색)
                if mar > THRESH_MAR:
                    cv2.putText(display_frame, "!!! YAWN DETECTED !!!", (50, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

                # 왼쪽 상단에 실시간 수치 표시 (디버깅용)
                cv2.putText(display_frame, f"EAR: {ear:.3f} MAR: {mar:.3f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Warning System", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
