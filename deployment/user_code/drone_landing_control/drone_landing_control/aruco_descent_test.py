import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

import cv2
import cv2.aruco as aruco
import numpy as np
import threading

# ==========================================
# 1. 캘리브레이션 데이터 세팅
# ==========================================
camera_matrix = np.array([
    [848.42845317,   0.        , 282.8888751 ],
    [  0.        , 847.1861915 , 250.95845253],
    [  0.        ,   0.        ,   1.        ]
])
dist_coeffs = np.array([
    [-2.96589206e-01,  1.47844047e+00,  2.67233193e-03, -5.57871408e-03, -5.36524818e+00]
])

MARKER_SIZE = 0.025  # 25mm = 0.025m

# [수정됨] 구버전(4.5.x) ArUco 문법으로 롤백!
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters_create()

# ==========================================
# 2. 비행 제어 + 녹화 통합 노드
# ==========================================
class ArucoDescentTestNode(Node):
    def __init__(self):
        super().__init__('aruco_descent_test_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz
        self.timer_count = 0
        
        self.target_altitude = 0.0 
        self.flight_state = "INIT"
        self.is_recording = True
        
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
            self.get_logger().info("3단계: 목표 고도 10m로 쾌속 상승!")
            self.target_altitude = -10.0 # 위로 10m
            self.flight_state = "CLIMBING"

        elif self.timer_count == 150: 
            self.get_logger().info("4단계: 10m 도달. 천천히 하강하며 마커를 탐색합니다 🚁⬇️")
            self.flight_state = "DESCENDING"

        if self.flight_state == "DESCENDING":
            if self.target_altitude < -0.5: 
                self.target_altitude += 0.05 # 초당 0.5m씩 천천히 하강
            else:
                self.get_logger().info("5단계: 지면 접근 완료. 착륙(Land) 및 녹화 종료 🛑")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.flight_state = "LANDED"
                self.is_recording = False

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [0.0, 0.0, self.target_altitude] 
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
    # 카메라 녹화 스레드
    # ------------------------------------------
    def camera_record_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('aruco_descent_10m.mp4', fourcc, 30.0, (640, 480))

        self.get_logger().info("🎥 카메라 녹화 시작: aruco_descent_10m.mp4")

        while self.is_recording:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # [수정됨] 구버전 문법 적용
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            cv2.putText(frame, f"Target Alt: {-self.target_altitude:.1f}m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                # [수정됨] solvePnP 대신 편하게 배열로 뽑아주는 estimatePoseSingleMarkers 사용
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                
                for i in range(len(ids)):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.015)
                    distance_z = tvecs[i][0][2]
                    text = f"Dist: {distance_z:.3f}m"
                    cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Marker LOST", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        self.get_logger().info("💾 녹화 완료 및 저장 성공!")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDescentTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.is_recording = False 
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()