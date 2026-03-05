import cv2
import time
import os

# 폴더 자동 생성 (에러 방지)
os.makedirs('./checkerboards', exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다. ROS2 노드가 꺼져있는지 확인하세요!")
    exit()

print("📸 3초 뒤부터 2초 간격으로 30장 자동 촬영을 시작합니다!")
print("카메라 앞에서 체커보드를 상하좌우, 멀리, 비스듬하게 계속 움직여주세요!")
time.sleep(3)

for i in range(30):
    ret, frame = cap.read()
    if ret:
        filename = f"./checkerboards/calib_{i+1:02d}.png"
        cv2.imwrite(filename, frame)
        print(f"[{i+1}/30] {filename} 찰칵! (자세 변경하세요~)")
    time.sleep(2) # 2초 대기

cap.release()
print("✅ 촬영 완료! ./checkerboards 폴더에 사진이 저장되었습니다.")