#http://192.168.43.135:5000/
import cv2
from flask import Flask, Response

app = Flask(__name__)
# 0번 카메라 연결 (안 되면 -1이나 1 등 다른 숫자로 변경)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # --- 나중에 이 부분에 ArUco 마커 인식 코드를 추가하시면 됩니다! ---
        # 예시: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #       corners, ids, rejected = cv2.aruco.detectMarkers(...)
        #       cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # -----------------------------------------------------------

        # 프레임을 JPEG 이미지로 변환
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # 웹브라우저로 스트리밍 송출
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 포트 번호 5000번으로 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=False)