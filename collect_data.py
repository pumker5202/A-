import cv2
import dlib
import numpy as np
import csv
from ultralytics import YOLO
from scipy.spatial import distance as dist

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = dist.euclidean(mouth[14], mouth[18]); B = dist.euclidean(mouth[13], mouth[19])
    C = dist.euclidean(mouth[12], mouth[16])
    return (A + B) / (2.0 * C)

model = YOLO("runs/detect/pure_eye_mouth_model/weights/best.pt")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

f = open('drowsy_data.csv', 'a', encoding='utf-8', newline='')
wr = csv.writer(f)

cap = cv2.VideoCapture(0)
print("--- 데이터 수집 중 (눈 박스 무관 감지) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, conf=0.5, verbose=False)
    key = cv2.waitKey(30) & 0xFF
    label = -1
    if key == ord('1'): label = 0
    elif key == ord('2'): label = 1
    elif key == ord('q'): break

    # r.plot() 결과물을 먼저 복사
    display_frame = frame.copy()

    for r in results:
        display_frame = r.plot() # YOLO 박스(Face 등) 그리기
        
        for box in r.boxes:
            # 핵심: 눈 박스가 사라져도 'Face(0번)' 박스만 있으면 dlib 실행!
            if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 얼굴 전체 영역을 dlib 사각형으로 변환
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                landmarks = predictor(frame, dlib_rect)
                pts = np.array([[p.x, p.y] for p in landmarks.parts()])

                # dlib은 얼굴 박스 내에서 눈 감음을 수치로 계산함
                ear = (calculate_ear(pts[36:42]) + calculate_ear(pts[42:48])) / 2.0
                mar = calculate_mar(pts[48:68])

                # 초록색 포인트 가시화 (이제 눈을 감아도 점들이 보일 겁니다)
                for (px, py) in pts:
                    cv2.circle(display_frame, (px, py), 1, (0, 255, 0), -1)

                if label != -1:
                    wr.writerow([ear, mar, label])
                    cv2.putText(display_frame, f"SAVED: {label}", (x1, y1-10), 1, 1.5, (0, 255, 0), 2)

                cv2.putText(display_frame, f"EAR: {ear:.2f} MAR: {mar:.2f}", (20, 40), 1, 0.7, (255, 255, 255), 2)

    cv2.imshow("Data Collector", display_frame)

cap.release(); f.close(); cv2.destroyAllWindows()
