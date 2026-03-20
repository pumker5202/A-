========================================
  DMS v6 (졸음감지) - 압축 해제 후 실행
========================================

1. 압축 해제
   tar -xzf dms_v6_board.tar.gz
   cd dms_v6_board

2. 실행 권한 (선택)
   chmod +x run.sh

3. 실행
   # 웹 모드 (브라우저 http://IP:5000)
   ./run.sh
   # 또는
   python3 dms_v6_eye_unified.py

   # 로컬 창 모드 (디스플레이 연결 시)
   ./run.sh --no-web
   # 또는
   python3 dms_v6_eye_unified.py --no-web

4. 종료
   로컬 창: q 키
   웹: Ctrl+C (터미널에서)

5. 필수 환경
   - Orange Pi 5+ (RK3588) 또는 RKNN 호환 보드
   - pip3 install opencv-python numpy dlib rknn-toolkit-lite2 flask
   - 웹캠 연결
