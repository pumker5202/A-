import cv2
import os
import time

# 1. 저장할 폴더 생성
base_dir = "local_dataset"
classes = ['bottle', 'human']
for cls in classes:
    os.makedirs(f"{base_dir}/{cls}", exist_ok=True)

# 2. 웹캠 시작
cap = cv2.VideoCapture(0)

print("📸 선우님, 데이터 수집을 시작합니다!")
print("'h'를 누르면 Human, 'b'를 누르면 Bottle 폴더에 저장됩니다. 'q'는 종료.")

count = {'human': 0, 'bottle': 0}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow('Collect Data - Sunwoo', frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'h' 누르면 사람 사진 저장
    if key == ord('h'):
        count['human'] += 1
        filename = f"{base_dir}/human/human_{count['human']}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Human 사진 저장 완료! (총 {count['human']}장)")
        
    # 'b' 누르면 병 사진 저장
    elif key == ord('b'):
        count['bottle'] += 1
        filename = f"{base_dir}/bottle/bottle_{count['bottle']}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Bottle 사진 저장 완료! (총 {count['bottle']}장)")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✨ 수집 완료! Human: {count['human']}장, Bottle: {count['bottle']}장")
print(f"📁 사진은 'local_dataset' 폴더에 저장되었습니다.")
