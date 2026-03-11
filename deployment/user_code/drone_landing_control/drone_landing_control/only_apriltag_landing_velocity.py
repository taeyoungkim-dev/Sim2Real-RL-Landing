import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition

import cv2
import numpy as np
import threading
import pupil_apriltags
import os
import datetime
import time

# ==========================================
# 1. 캘리브레이션 및 AprilTag 설정
# ==========================================
camera_matrix = np.array([
    [848.42845317,   0.        , 282.8888751 ],
    [  0.        , 847.1861915 , 250.95845253],
    [  0.        ,   0.        ,   1.        ]
])
dist_coeffs = np.array([
    [-2.96589206e-01,  1.47844047e+00,  2.67233193e-03, -5.57871408e-03, -5.36524818e+00]
])

MARKER_SIZE = 0.3

# pupil-apriltags 카메라 파라미터: [fx, fy, cx, cy]
camera_params = [
    camera_matrix[0, 0],  # fx
    camera_matrix[1, 1],  # fy
    camera_matrix[0, 2],  # cx
    camera_matrix[1, 2],  # cy
]

apriltag_detector = pupil_apriltags.Detector(families='tag36h11')

# ==========================================
# 2. 비행 제어 + 혼합 제어 (위치/속도) 노드
# ==========================================
class AprilTagUltimateLandNode(Node):
    def __init__(self):
        super().__init__('apriltag_ultimate_land_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publisher
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # Subscriber (Odometry 수신용)
        self.local_pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_callback, qos)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0

        # 드론의 현재 실제 위치 변수 (Odometry)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0  # NED 좌표계 (예: -7.0 이 7m 상공)

        # 마커를 놓쳤을 때 위치를 고정할 변수 (제자리 하강용)
        self.hold_x = 0.0
        self.hold_y = 0.0

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.flight_state = "INIT"
        self.is_recording = True   # 카메라 루프 제어 (NAV_LAND + 5초 후 False)

        # 비행 파라미터 (NED Z)
        self.takeoff_target_z = -5.0
        self.descend_switch_z = self.takeoff_target_z + 0.5  # -4.5m
        self.land_trigger_z = -0.5
        self.takeoff_start_tick = 50
        self.target_altitude = 0.0

        # CLIMBING 진입 시각 (하강 전환 시간 조건용)
        self.climbing_start_time = None
        self.descend_min_elapsed_sec = 5.0  # 상승 명령 후 최소 경과 시간 (GPS 튐 방지)

        # NAV_LAND 명령 시각 (착륙 후 추가 녹화용)
        self.landing_time = None
        self.post_land_record_sec = 5.0  # NAV_LAND 신호 후 추가 녹화 시간

        # 칼만 필터 초기 세팅
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0.033, 0], [0, 1, 0, 0.033], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf_initialized = False

        self.marker_visible = False
        self.filtered_x = 0.0
        self.filtered_y = 0.0

        self.cam_thread = threading.Thread(target=self.camera_record_loop)
        self.cam_thread.start()

    # Odometry 콜백 함수
    def local_pos_callback(self, msg):
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.timer_count == 20:
            self.get_logger().info("1단계: Offboard 모드 진입!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        elif self.timer_count == 30:
            self.get_logger().info("2단계: 시동(Arm) 완료")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.flight_state = "ARMED_WAIT"
            self.target_altitude = 0.0
            self.get_logger().info("3단계: 모터 안정화 대기 (30틱~50틱), 고도 0.0m 유지")

        elif self.timer_count == self.takeoff_start_tick and self.flight_state in ["INIT", "ARMED_WAIT"]:
            self.flight_state = "CLIMBING"
            self.target_altitude = self.takeoff_target_z
            self.climbing_start_time = self.get_clock().now()
            self.get_logger().info(f"4단계: 목표 고도 5m로 부드럽게 상승 시작!")

        elif self.flight_state == "CLIMBING":
            # 하강 전환 조건: 고도 도달 OR 상승 명령 후 최소 경과 시간 충족
            # GPS 튐으로 고도 조건이 일시적으로 충족되더라도 시간 조건으로 안전하게 전환 가능
            elapsed_sec = (self.get_clock().now() - self.climbing_start_time).nanoseconds / 1e9
            altitude_reached = self.current_z <= self.descend_switch_z
            time_elapsed = elapsed_sec >= self.descend_min_elapsed_sec

            if altitude_reached or time_elapsed:
                self.flight_state = "DESCENDING"
                trigger = "고도" if altitude_reached else "경과시간"
                self.get_logger().info(
                    f"5단계: [{trigger}] 조건 충족 (고도={self.current_z:.1f}m, 경과={elapsed_sec:.1f}s). 마커 추적 하강 시작"
                )
                # 하강을 시작할 때, 기본 위치를 현재 위치로 락온(Lock-on)
                self.hold_x = self.current_x
                self.hold_y = self.current_y

        elif self.flight_state == "DESCENDING":
            # 실제 고도가 0.5m (NED -0.5) 미만으로 내려왔을 때 착륙
            if self.current_z > self.land_trigger_z:
                self.get_logger().info("6단계: 지면 접근 완료. 자동 착륙(Land)")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.flight_state = "LANDED"
                self.landing_time = time.time()  # 착륙 신호 시각 기록 (카메라는 5초 더 녹화)
            else:
                if self.marker_visible and self.kf_initialized:
                    Kp_vel = 0.8
                    self.target_vx = float(np.clip(Kp_vel * (-self.filtered_y), -0.5, 0.5))
                    self.target_vy = float(np.clip(Kp_vel * (self.filtered_x), -0.5, 0.5))

                    # 마커가 보일 때는 현재 내 위치를 계속 업데이트 (놓쳤을 때 대비)
                    self.hold_x = self.current_x
                    self.hold_y = self.current_y
                else:
                    self.target_vx = 0.0
                    self.target_vy = 0.0

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True  # 위치와 속도 모두 제어 허용
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()

        if self.flight_state in ["INIT", "ARMED_WAIT", "CLIMBING"]:
            # INIT/대기 구간은 0.0m 유지, 50틱 이후에 목표 고도로 전환
            msg.position = [0.0, 0.0, self.target_altitude]
            msg.velocity = [np.nan, np.nan, np.nan]

        elif self.flight_state == "DESCENDING":
            if self.marker_visible:
                # 마커 발견 시: X, Y, Z 모두 속도(Velocity) 제어
                msg.position = [np.nan, np.nan, np.nan]
                msg.velocity = [self.target_vx, self.target_vy, 0.8]  # Vz=0.8 (하강)
            else:
                # 마커 놓쳤을 때: X, Y는 위치(Position) 고정, Z만 속도(Velocity) 하강
                msg.position = [self.hold_x, self.hold_y, np.nan]
                msg.velocity = [np.nan, np.nan, 0.8]  # Vz=0.8 (하강)

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
    # 카메라 녹화 및 칼만 필터 스레드
    # ------------------------------------------
    def camera_record_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 동영상 저장 경로: 패키지 루트/video/찍은시각.mp4
        video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'video')
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(video_dir, f'{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        self.get_logger().info(f"카메라 녹화 시작: {video_path}")

        # undistort에 사용할 0 왜곡 계수 (프레임을 펴고 나면 왜곡이 없는 상태)
        zero_dist = np.zeros((4, 1))

        # 왜곡 보정 맵을 루프 전에 한 번만 계산 (remap 최적화)
        # cv2.undistort를 매 프레임 호출하면 픽셀마다 왜곡 방정식을 재계산하지만,
        # initUndistortRectifyMap으로 룩업 테이블을 미리 만들어두면 remap이 단순 참조만 수행
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (640, 480), cv2.CV_16SC2
        )

        while self.is_recording:
            ret, frame = cap.read()
            if not ret:
                continue

            # 렌즈 왜곡 보정: 미리 계산된 맵으로 빠르게 리매핑
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = apriltag_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=MARKER_SIZE
            )

            # 현재 드론의 실제 고도(Odometry) 표시
            cv2.putText(frame, f"Real Alt: {-self.current_z:.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if self.kf_initialized:
                prediction = self.kf.predict()
                self.filtered_x = prediction[0][0]
                self.filtered_y = prediction[1][0]

            if len(detections) > 0:
                self.marker_visible = True
                detection = detections[0]

                # AprilTag 코너 및 ID 시각화
                corners = detection.corners.astype(int)
                cv2.polylines(frame, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                cv2.putText(frame, str(detection.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                raw_x = detection.pose_t[0][0]
                raw_y = detection.pose_t[1][0]

                measurement = np.array([[raw_x], [raw_y]], dtype=np.float32)

                if not self.kf_initialized:
                    self.kf.statePost = np.array([[raw_x], [raw_y], [0], [0]], dtype=np.float32)
                    self.kf_initialized = True
                else:
                    estimated = self.kf.correct(measurement)
                    self.filtered_x = estimated[0][0]
                    self.filtered_y = estimated[1][0]

                # 회전 행렬을 회전 벡터로 변환하여 좌표축 시각화
                # 프레임이 이미 undistort 처리되었으므로 zero_dist 사용
                rvec, _ = cv2.Rodrigues(detection.pose_R)
                tvec = detection.pose_t
                cv2.drawFrameAxes(frame, camera_matrix, zero_dist, rvec, tvec, MARKER_SIZE/2)

                text = f"Mode: TRACKING | Vx:{self.target_vx:.2f} Vy:{self.target_vy:.2f}"
                cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.circle(frame, (int(320 + raw_x * 1000), int(240 + raw_y * 1000)), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(320 + self.filtered_x * 1000), int(240 + self.filtered_y * 1000)), 8, (0, 255, 0), 2)

            else:
                self.marker_visible = False
                if self.kf_initialized:
                    cv2.putText(frame, "Mode: HOLD XY & DESCEND Z", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.circle(frame, (int(320 + self.filtered_x * 1000), int(240 + self.filtered_y * 1000)), 8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Marker LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            height, width = frame.shape[:2]
            cv2.line(frame, (int(width / 2) - 20, int(height / 2)), (int(width / 2) + 20, int(height / 2)), (255, 0, 0), 2)
            cv2.line(frame, (int(width / 2), int(height / 2) - 20), (int(width / 2), int(height / 2) + 20), (255, 0, 0), 2)

            out.write(frame)

            # NAV_LAND 신호 후 post_land_record_sec 초가 지나면 루프 종료
            if self.landing_time is not None and \
               time.time() - self.landing_time >= self.post_land_record_sec:
                self.is_recording = False

        cap.release()
        out.release()
        self.get_logger().info(f"녹화 완료 및 저장 성공! ({video_path})")


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagUltimateLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.is_recording = False
        node.cam_thread.join()  # 카메라 스레드가 완전히 종료될 때까지 대기
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
