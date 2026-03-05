import cv2
import numpy as np
import glob

# 1. 체커보드 설정 (엔지니어님 팩트 체크 완료 스펙)
CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.025  # 25mm = 0.025m

# 2. 3D 실제 좌표(objp) 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] # 3D 실제 세계의 점
imgpoints = [] # 2D 사진 상의 점

# 3. 사진 불러오기
images = glob.glob('./checkerboards/*.png')
if not images:
    print("사진을 찾을 수 없습니다! ./checkerboards 폴더 안에 사진이 있는지 확인해주세요.")
    exit()

print(f"총 {len(images)}장의 사진으로 캘리브레이션을 시작합니다. (시간이 조금 걸릴 수 있습니다...)")

# 사진 속 코너를 더 정밀하게 찾는 옵션
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

success_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        # 코너 위치를 픽셀 이하 단위로 초정밀 조정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)
        success_count += 1
        print(f"[{success_count}] {fname} - 패턴 인식 성공!")
    else:
        print(f"[실패] {fname} - 패턴을 찾지 못했습니다. (사진이 흔들렸거나 덜 찍혔을 수 있습니다)")

if success_count == 0:
    print("패턴을 인식한 사진이 단 한 장도 없습니다! 사진을 다시 찍어야 합니다.")
    exit()

print(f"\n성공적으로 패턴을 찾은 사진 수: {success_count}/{len(images)}장")
print("수학적 매트릭스 계산 중... ⏳\n")

# 4. 대망의 카메라 매트릭스 계산!
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("="*60)
print("🎯 [캘리브레이션 완료! 결과 데이터]")
print("="*60)
print("\n1. Camera Matrix (mtx):")
print(mtx)
print("\n2. Distortion Coefficients (dist):")
print(dist)
print("="*60)

# 나중을 위해 파일로도 저장해둡니다.
np.savez('calib_data.npz', mtx=mtx, dist=dist)
print("\n✅ 현재 폴더에 'calib_data.npz' 파일로도 안전하게 저장되었습니다!")