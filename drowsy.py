import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import math
from collections import deque

# ===============================================
# 1. 학습한 모델 구조 불러오기
# ===============================================
class LSTMDrowsy(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

device = torch.device("cpu") # 가벼운 CPU 모드
print(f"사용 중인 기기: {device}")

model = LSTMDrowsy().to(device)
model.load_state_dict(torch.load('lstm_drowsy_v2.pth', map_location=device, weights_only=True))
model.eval()

# ===============================================
# 2. MediaPipe 세팅 (눈 깜빡임 + 고개 끄덕임)
# ===============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

def calculate_ear(eye_points, landmarks, img_w, img_h):
    coords = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in eye_points]
    v1 = math.dist(coords[1], coords[5])
    v2 = math.dist(coords[2], coords[4])
    h = math.dist(coords[0], coords[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

# ===============================================
# 3. 실시간 웹캠 연결 (핵심 부분!)
# ===============================================
# 0번은 우분투에 연결된 기본 웹캠을 의미합니다.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 찾을 수 없습니다! 우분투에 카메라가 연결되어 있는지 확인해주세요.")
    exit()

sequence_data = deque(maxlen=10)
print("🎥 웹캠이 켜졌습니다! 화면을 끄려면 영문 상태에서 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "Tracking..."
    color = (255, 255, 255)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 특징 추출
        l_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
        r_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
        avg_ear = (l_ear + r_ear) / 2.0
        nose_x = landmarks[NOSE_TIP].x
        nose_y = landmarks[NOSE_TIP].y

        # 프레임 단위로 데이터 큐에 쌓기
        sequence_data.append([avg_ear, nose_x, nose_y])

        # 10프레임이 모이면 판단 시작!
        if len(sequence_data) == 10:
            input_tensor = torch.tensor([list(sequence_data)], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(input_tensor).item()
            
            # 확률값이 0.5 이상이면 졸음으로 판정
            if output > 0.5:
                status_text = f"DROWSY!!! ({output:.2f})"
                color = (0, 0, 255) # BGR 기준: 빨간색
            else:
                status_text = f"AWAKE ({output:.2f})"
                color = (0, 255, 0) # BGR 기준: 초록색

    # 화면에 글자 입히기
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # 내 얼굴 화면 띄우기
    cv2.imshow('Real-time Drowsiness Detection', frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
