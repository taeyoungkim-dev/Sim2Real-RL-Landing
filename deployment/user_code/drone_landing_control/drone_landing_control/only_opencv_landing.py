import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

import cv2
import cv2.aruco as aruco
import numpy as np
import threading

# ==========================================
# 1. 캘리브레이션 및 ArUco 설정
# ==========================================
camera_matrix = np.array([
    [848.42845317,   0.        , 282.8888751 ],
    [  0.        , 847.1861915 , 250.95845253],
    [  0.        ,   0.        ,   1.        ]
])
dist_coeffs = np.array([
    [-2.96589206e-01,  1.47844047e+00,  2.67233193e-03, -5.57871408e-03, -5.36524818e+00]
])

MARKER_SIZE = 0.025

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters_create()

# ==========================================
# 2. 비행 제어 + 칼만 필터 시각 추적 노드
# ==========================================
class ArucoTrackingLandNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_land_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0
        
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_altitude = 0.0 
        self.flight_state = "INIT"
        self.is_recording = True
        
        # 🌟 칼만 필터 초기 세팅 (상태 변수 4개: x, y, vx, vy / 측정 변수 2개: x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # dt = 0.033 (약 30fps 카메라 기준)
        self.kf.transitionMatrix = np.array([[1, 0, 0.033, 0], [0, 1, 0, 0.033], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 노이즈 튜닝 (값이 작을수록 센서보다 이전 궤적을 믿음)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3     # 모델(관성) 노이즈
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2 # 카메라(센서) 노이즈
        self.kf_initialized = False

        self.marker_visible = False
        self.filtered_x = 0.0 # 버터처럼 부드러워진 X 오차
        self.filtered_y = 0.0 # 버터처럼 부드러워진 Y 오차

        self.cam_thread = threading.Thread(target=self.camera_record_loop)
        self.cam_thread.start()

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.timer_count == 20:
            self.get_logger().info("1단계: Offboard 모드 진입!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        elif self.timer_count == 30:
            self.get_logger().info("2단계: 시동(Arm) 완료 🚀")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        elif self.timer_count == 50:
            self.get_logger().info("3단계: 목표 고도 10m로 상승!")
            self.target_altitude = -10.0
            self.flight_state = "CLIMBING"

        elif self.timer_count == 150: 
            self.get_logger().info("4단계: 10m 도달. 마커 추적 하강 시작 (칼만 필터 ON) 🚁⬇️🎯")
            self.flight_state = "DESCENDING"

        if self.flight_state == "DESCENDING":
            # 1. 고도 하강
            if self.target_altitude < -0.5: 
                self.target_altitude += 0.05 
            else:
                self.get_logger().info("5단계: 착륙 고도 도달. 자동 착륙(Land) 🛑")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.flight_state = "LANDED"
                self.is_recording = False

            # 2. X, Y 위치 조정 (Raw 데이터가 아닌 '칼만 필터의 예측값' 사용)
            if self.kf_initialized:
                # 필터 덕분에 급발진이 없으므로 P게인을 조금 높여도 안정적입니다
                Kp = 0.25 
                
                # 카메라 프레임(X:오른쪽, Y:아래) -> 드론 NED 프레임(X:앞, Y:오른쪽) 변환
                move_x = Kp * (-self.filtered_y)
                move_y = Kp * (self.filtered_x)
                
                move_x = np.clip(move_x, -0.2, 0.2)
                move_y = np.clip(move_y, -0.2, 0.2)
                
                self.target_x += move_x
                self.target_y += move_y

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [self.target_x, self.target_y, self.target_altitude] 
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1, msg.param2 = param1, param2
        msg.target_system, msg.target_component = 1, 1
        msg.source_system, msg.source_component = 1, 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    # ------------------------------------------
    # 카메라 녹화 및 칼만 필터 처리 스레드
    # ------------------------------------------
    def camera_record_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('aruco_kalman_land.mp4', fourcc, 30.0, (640, 480))

        self.get_logger().info("🎥 칼만 필터 추적 녹화 시작: aruco_kalman_land.mp4")

        while self.is_recording:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            cv2.putText(frame, f"Alt: {-self.target_altitude:.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Target X:{self.target_x:.2f} Y:{self.target_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 🌟 1. 칼만 필터 '예측(Predict)' 단계: 마커가 안 보여도 관성으로 위치를 예측함
            if self.kf_initialized:
                prediction = self.kf.predict()
                self.filtered_x = prediction[0][0]
                self.filtered_y = prediction[1][0]

            if ids is not None:
                self.marker_visible = True
                aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                
                raw_x = tvecs[0][0][0]
                raw_y = tvecs[0][0][1]
                distance_z = tvecs[0][0][2]
                
                # 🌟 2. 칼만 필터 '업데이트(Correct)' 단계: 실제 관측된 데이터 입력
                measurement = np.array([[raw_x], [raw_y]], dtype=np.float32)
                
                if not self.kf_initialized:
                    # 첫 발견 시 칼만 필터의 초기값을 현재 위치로 강제 설정 (초기 급발진 방지)
                    self.kf.statePost = np.array([[raw_x], [raw_y], [0], [0]], dtype=np.float32)
                    self.kf_initialized = True
                else:
                    estimated = self.kf.correct(measurement)
                    self.filtered_x = estimated[0][0]
                    self.filtered_y = estimated[1][0]
                
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.015)
                
                text = f"Dist: {distance_z:.3f}m | KF_X:{self.filtered_x:.2f} KF_Y:{self.filtered_y:.2f}"
                cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 원본 노이즈 데이터와 부드러운 필터 데이터를 비교하는 UI 점 찍기
                cv2.circle(frame, (int(320 + raw_x*1000), int(240 + raw_y*1000)), 5, (0, 0, 255), -1) # 빨간점: 통통 튀는 Raw 데이터
                cv2.circle(frame, (int(320 + self.filtered_x*1000), int(240 + self.filtered_y*1000)), 8, (0, 255, 0), 2) # 초록원: 부드러운 Kalman 데이터

            else:
                self.marker_visible = False
                if self.kf_initialized:
                    cv2.putText(frame, "Marker LOST - Predicting...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    # 예측된 위치에 노란색 원 그리기
                    cv2.circle(frame, (int(320 + self.filtered_x*1000), int(240 + self.filtered_y*1000)), 8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Marker LOST", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 중앙 십자선
            height, width = frame.shape[:2]
            cv2.line(frame, (int(width/2) - 20, int(height/2)), (int(width/2) + 20, int(height/2)), (255, 0, 0), 2)
            cv2.line(frame, (int(width/2), int(height/2) - 20), (int(width/2), int(height/2) + 20), (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        self.get_logger().info("💾 녹화 완료 및 저장 성공!")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoTrackingLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.is_recording = False 
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()