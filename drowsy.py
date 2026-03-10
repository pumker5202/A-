import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import math

# ===============================================
# 1. 새롭게 진화한 순간포착 뇌 (DNN) 구조
# ===============================================
class DNNDrowsy(nn.Module):
    def __init__(self, input_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cpu")
print(f"사용 중인 기기: {device}")

# ★ 새로 학습한 v3 뇌(가중치) 불러오기
model = DNNDrowsy().to(device)
model.load_state_dict(torch.load('dnn_drowsy_v3.pth', map_location=device, weights_only=True))
model.eval()

# ===============================================
# 2. MediaPipe 세팅 (눈 깜빡임 + 코 좌표)
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
# 3. 실시간 웹캠 분석 시작!
# ===============================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 찾을 수 없습니다!")
    exit()

print("🎥 V3 웹캠이 켜졌습니다! 화면을 끄려면 영문 상태에서 'q'를 누르세요.")

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
        
        # 특징 3가지 추출 (EAR, 코X, 코Y)
        l_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
        r_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
        avg_ear = (l_ear + r_ear) / 2.0
        nose_x = landmarks[NOSE_TIP].x
        nose_y = landmarks[NOSE_TIP].y

        # ★ V3의 핵심: 10프레임 기다릴 필요 없이 즉시 AI에게 전달!
        input_tensor = torch.tensor([[avg_ear, nose_x, nose_y]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor).item()
        
        # 확률값이 0.5 이상이면 졸음으로 판정
        if output > 0.5:
            status_text = f"DROWSY!!! ({output:.2f})"
            color = (0, 0, 255) # 빨간색
        else:
            status_text = f"AWAKE ({output:.2f})"
            color = (0, 255, 0) # 초록색

    # 화면에 상태 표시
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    cv2.imshow('V3 Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
