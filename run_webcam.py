import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# GPU 없이 CPU로만 실행하도록 설정
device = torch.device("cpu")

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.6),
    nn.Linear(1280, 2)
)

# 모델 로드
model.load_state_dict(torch.load("best_recycling_model.pt", map_location=device))
model.eval()

# 전처리 (학습 때와 동일)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Bottle', 'Human']
cap = cv2.VideoCapture(0)

print("📸 [CPU 모드] 웹캠 테스트 시작! 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds.item()]

    # 결과 표시
    cv2.putText(frame, f"Result: {label}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow('Recycling Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
