import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition

import cv2
import numpy as np
import threading
import time
import pupil_apriltags
import os
import datetime

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

camera_params = [
    camera_matrix[0, 0],  # fx
    camera_matrix[1, 1],  # fy
    camera_matrix[0, 2],  # cx
    camera_matrix[1, 2],  # cy
]

apriltag_detector = pupil_apriltags.Detector(families='tag36h11')

# ==========================================
# 2. 비행 제어 노드 (순수 속도 제어)
# ==========================================
class FinalAprilTagLandNode(Node):
    def __init__(self):
        super().__init__('final_apriltag_land_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # 초기 Heading 캡처 (body↔NED 변환용, 한 번만 수신)
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_pos_callback, qos
        )

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0

        # --------------------------------------------------
        # 비행 상태 및 속도 명령 (NED 프레임)
        # --------------------------------------------------
        self.flight_state = "INIT"
        self.stop_recording_event = threading.Event()

        self.cmd_vx = 0.0  # NED X (North)
        self.cmd_vy = 0.0  # NED Y (East)
        self.cmd_vz = 0.0  # NED Z (Down, 양수=하강)

        # --------------------------------------------------
        # 초기 Yaw (body 프레임 속도를 NED로 변환)
        # --------------------------------------------------
        self.initial_yaw = None

        # --------------------------------------------------
        # CLIMBING 타이머
        # --------------------------------------------------
        self.climbing_start_time = None
        self.climbing_duration_sec = 14.0  # velocity [0,0,-1]을 14초간 유지

        # --------------------------------------------------
        # SEARCHING - 구역: 드론 전방 기준 좌3m~우3m × 전방5m (6×5 직사각형)
        # Lawnmower 패턴 (body 프레임: vx=전방, vy=우측, 음수=좌측)
        #
        # 줄 위치(body Y): -3, -1.5, 0, +1.5, +3  (1.5m 간격 5줄)
        # 이동 경로:
        #   좌측 3m → 전방 5m → 우 1.5m → 후방 5m → 우 1.5m → 전방 5m → ...
        # --------------------------------------------------
        _S = 0.5  # 탐색 속도 (m/s)
        self.search_plan = [
            # (vx_body, vy_body, duration_s)
            (0.0, -_S, 6.0),   # 좌측 3m  (0 → -3)
            (_S,  0.0, 10.0),  # 전방 5m  (strip y=-3)
            (0.0,  _S, 3.0),   # 우측 1.5m (-3 → -1.5)
            (-_S, 0.0, 10.0),  # 후방 5m  (strip y=-1.5)
            (0.0,  _S, 3.0),   # 우측 1.5m (-1.5 → 0)
            (_S,  0.0, 10.0),  # 전방 5m  (strip y=0)
            (0.0,  _S, 3.0),   # 우측 1.5m (0 → +1.5)
            (-_S, 0.0, 10.0),  # 후방 5m  (strip y=+1.5)
            (0.0,  _S, 3.0),   # 우측 1.5m (+1.5 → +3)
            (_S,  0.0, 10.0),  # 전방 5m  (strip y=+3)
        ]
        self.search_phase_idx = 0
        self.search_phase_start_time = None

        # AprilTag 최초 발견 여부 (True이면 lawnmower → 정렬 모드 전환)
        self.tag_found_ever = False


        # --------------------------------------------------
        # AprilTag 추적 속도 (SEARCHING 정렬 & DESCENDING 공용)
        # 태그를 놓쳤을 때도 마지막 계산값을 유지
        # --------------------------------------------------
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.descending_velocity = 0.7

        # --------------------------------------------------
        # DESCENDING - 태그 소실 타이머 (0.5초 소실 시 LANDING)
        # --------------------------------------------------
        self.tag_lost_start_time = None
        self.tag_lost_threshold_sec = 0.5

        # --------------------------------------------------
        # 칼만 필터
        # --------------------------------------------------
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 0.033, 0], [0, 1, 0, 0.033], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf_initialized = False

        self.marker_visible = False
        self.filtered_x = 0.0
        self.filtered_y = 0.0


        self.cam_thread = threading.Thread(target=self.camera_record_loop)
        self.cam_thread.start()

    # --------------------------------------------------
    # 초기 Heading 콜백 (한 번만 캡처)
    # --------------------------------------------------
    def local_pos_callback(self, msg):
        if self.initial_yaw is None:
            self.initial_yaw = msg.heading
            self.get_logger().info(f"초기 Yaw 캡처: {np.degrees(self.initial_yaw):.1f}°")

    def body_to_ned(self, vx_body, vy_body):
        """드론 바디 프레임 속도 → NED 속도 변환 (초기 Yaw 기준)"""
        yaw = self.initial_yaw if self.initial_yaw is not None else 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        return c * vx_body - s * vy_body, s * vx_body + c * vy_body

    # --------------------------------------------------
    # 메인 타이머 (0.1s 주기)
    # --------------------------------------------------
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
            self.cmd_vx, self.cmd_vy, self.cmd_vz = 0.0, 0.0, 0.0
            self.get_logger().info("3단계: 모터 안정화 대기. velocity [0, 0, 0]")

        elif self.timer_count == 50 and self.flight_state in ["INIT", "ARMED_WAIT"]:
            self.flight_state = "CLIMBING"
            self.climbing_start_time = self.get_clock().now()
            self.cmd_vx, self.cmd_vy, self.cmd_vz = 0.0, 0.0, -1.0
            self.get_logger().info("4단계: CLIMBING 시작. velocity [0, 0, -1], 10초")

        elif self.flight_state == "CLIMBING":
            elapsed = (self.get_clock().now() - self.climbing_start_time).nanoseconds / 1e9
            if elapsed >= self.climbing_duration_sec:
                self.flight_state = "SEARCHING"
                self.search_phase_idx = 0
                self.search_phase_start_time = self.get_clock().now()
                vx_b, vy_b, _ = self.search_plan[0]
                vx_ned, vy_ned = self.body_to_ned(vx_b, vy_b)
                self.cmd_vx, self.cmd_vy, self.cmd_vz = vx_ned, vy_ned, 0.0
                self.get_logger().info(
                    f"5단계: SEARCHING 시작. 탐색 패턴 {len(self.search_plan)}단계 (6×5m 구역)"
                )

        elif self.flight_state == "SEARCHING":
            self._handle_searching()

        elif self.flight_state == "DESCENDING":
            self._handle_descending()

        elif self.flight_state == "LANDING":
            self.cmd_vx, self.cmd_vy, self.cmd_vz = 0.0, 0.0, 0.0

        self.timer_count += 1

    # --------------------------------------------------
    # SEARCHING 상태 처리
    # --------------------------------------------------
    def _handle_searching(self):
        if self.tag_found_ever:
            # [접근 모드] 태그 방향으로 0.3 m/s 이동. 태그를 잃어도 lawnmower로 돌아가지 않음.
            if self.marker_visible and self.kf_initialized:
                # 태그 보임 → 오차 비례 속도 갱신 (최대 0.3 m/s)
                Kp = 0.8
                self.target_vx = float(np.clip(Kp * (-self.filtered_y), -0.2, 0.2))
                self.target_vy = float(np.clip(Kp * (self.filtered_x), -0.2, 0.2))
            # 태그 소실 시에도 마지막 계산 속도를 그대로 유지 (재탐색 없음)

            # KF 예측값으로 오차 계산 (태그 순간 소실 시에도 예측값 활용)
            if self.kf_initialized:
                error = np.sqrt(self.filtered_x ** 2 + self.filtered_y ** 2)
                if error < 0.5:
                    self.flight_state = "DESCENDING"
                    self.tag_lost_start_time = None
                    self.get_logger().info(
                        f"6단계: 오차 {error:.2f}m < 0.5m. DESCENDING 전환"
                    )

            self.cmd_vx = self.target_vx
            self.cmd_vy = self.target_vy
            self.cmd_vz = 0.0

        else:
            # [Lawnmower 탐색 모드]
            elapsed = (self.get_clock().now() - self.search_phase_start_time).nanoseconds / 1e9
            _, _, phase_dur = self.search_plan[self.search_phase_idx]

            if elapsed >= phase_dur:
                next_idx = self.search_phase_idx + 1
                if next_idx >= len(self.search_plan):
                    self.get_logger().info("탐색 패턴 완료. AprilTag 미발견. 착륙 명령")
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.flight_state = "LANDING"
                    return
                else:
                    self.search_phase_idx = next_idx
                    self.search_phase_start_time = self.get_clock().now()
                    vx_b, vy_b, dur = self.search_plan[next_idx]
                    vx_ned, vy_ned = self.body_to_ned(vx_b, vy_b)
                    self.cmd_vx, self.cmd_vy, self.cmd_vz = vx_ned, vy_ned, 0.0
                    self.get_logger().info(
                        f"탐색 {next_idx + 1}/{len(self.search_plan)}: "
                        f"NED vel=({vx_ned:.2f}, {vy_ned:.2f}, 0), 지속={dur:.0f}s"
                    )

            # 이번 틱에 태그를 처음 발견했을 경우 즉시 정렬 모드 전환 준비
            if self.marker_visible:
                self.tag_found_ever = True
                if self.kf_initialized:
                    Kp = 0.8
                    self.target_vx = float(np.clip(Kp * (-self.filtered_y), -1.0, 1.0))
                    self.target_vy = float(np.clip(Kp * (self.filtered_x), -1.0, 1.0))
                self.cmd_vx = self.target_vx
                self.cmd_vy = self.target_vy
                self.cmd_vz = 0.0
                self.get_logger().info("AprilTag 최초 발견! 정렬 모드 전환")

    # --------------------------------------------------
    # DESCENDING 상태 처리
    # --------------------------------------------------
    def _handle_descending(self):
        if self.marker_visible and self.kf_initialized:
            self.tag_lost_start_time = None
            Kp = 0.8
            self.target_vx = float(np.clip(Kp * (-self.filtered_y), -0.1, 0.1))
            self.target_vy = float(np.clip(Kp * (self.filtered_x), -0.1, 0.1))
        else:
            if self.tag_lost_start_time is None:
                self.tag_lost_start_time = self.get_clock().now()
            else:
                elapsed_lost = (self.get_clock().now() - self.tag_lost_start_time).nanoseconds / 1e9
                if elapsed_lost >= self.tag_lost_threshold_sec:
                    self.get_logger().info(
                        f"7단계: AprilTag {elapsed_lost:.1f}초 소실. 자동 착륙(Land)"
                    )
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.flight_state = "LANDING"

        self.cmd_vx = self.target_vx
        self.cmd_vy = self.target_vy
        self.cmd_vz = self.descending_velocity  # 양수 = NED 하강

    # --------------------------------------------------
    # PX4 메시지 발행
    # --------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [np.nan, np.nan, np.nan]
        msg.velocity = [self.cmd_vx, self.cmd_vy, self.cmd_vz]
        msg.yaw = float('nan')  # yaw 제어 미사용 (PX4가 현재 yaw 유지)
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

    # --------------------------------------------------
    # 카메라 녹화 및 칼만 필터 스레드
    # --------------------------------------------------
    def camera_record_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        video_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'video'
        )
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(video_dir, f'{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        self.get_logger().info(f"카메라 녹화 시작: {video_path}")

        zero_dist = np.zeros((4, 1))
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, (640, 480), cv2.CV_16SC2
        )

        while not self.stop_recording_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    # 카메라 프레임 읽기 실패 시에도 루프를 유지한다.
                    time.sleep(0.01)
                    continue

                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = apriltag_detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=MARKER_SIZE
                )

                # 비행 상태 오버레이
                state_colors = {
                    "INIT": (200, 200, 200), "ARMED_WAIT": (200, 200, 0),
                    "CLIMBING": (0, 200, 255), "SEARCHING": (255, 165, 0),
                    "DESCENDING": (0, 255, 0), "LANDING": (0, 0, 255),
                }
                color = state_colors.get(self.flight_state, (255, 255, 255))
                cv2.putText(frame, f"State: {self.flight_state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # KF 예측 (매 프레임)
                if self.kf_initialized:
                    prediction = self.kf.predict()
                    self.filtered_x = prediction[0][0]
                    self.filtered_y = prediction[1][0]

                if len(detections) > 0:
                    self.marker_visible = True
                    detection = detections[0]

                    corners = detection.corners.astype(int)
                    cv2.polylines(frame, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                    cv2.putText(frame, str(detection.tag_id), tuple(corners[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

                    rvec, _ = cv2.Rodrigues(detection.pose_R)
                    tvec = detection.pose_t
                    cv2.drawFrameAxes(frame, camera_matrix, zero_dist, rvec, tvec, MARKER_SIZE / 2)

                    error = np.sqrt(self.filtered_x ** 2 + self.filtered_y ** 2)
                    text = (f"TRACKING | Err:{error:.2f}m | "
                            f"Vx:{self.cmd_vx:.2f} Vy:{self.cmd_vy:.2f} Vz:{self.cmd_vz:.2f}")
                    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    cv2.circle(frame, (int(320 + raw_x * 1000), int(240 + raw_y * 1000)), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (int(320 + self.filtered_x * 1000), int(240 + self.filtered_y * 1000)), 8, (0, 255, 0), 2)

                else:
                    self.marker_visible = False
                    if self.tag_found_ever:
                        text = (f"Tag LOST | HOLD VEL | "
                                f"Vx:{self.cmd_vx:.2f} Vy:{self.cmd_vy:.2f} Vz:{self.cmd_vz:.2f}")
                        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                    elif self.flight_state == "SEARCHING":
                        phase_text = f"LAWNMOWER {self.search_phase_idx + 1}/{len(self.search_plan)}"
                        cv2.putText(frame, phase_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "No Tag", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

                    if self.kf_initialized:
                        cv2.circle(frame, (int(320 + self.filtered_x * 1000), int(240 + self.filtered_y * 1000)), 8, (0, 255, 255), 2)

                h, w = frame.shape[:2]
                cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (255, 0, 0), 2)
                cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (255, 0, 0), 2)

                out.write(frame)
            except Exception as exc:
                # 예외가 발생해도 Ctrl+C 전까지 녹화 루프를 유지한다.
                self.get_logger().error(f"camera_record_loop 예외 발생: {exc}")
                time.sleep(0.05)

        cap.release()
        out.release()
        self.get_logger().info(f"녹화 완료 및 저장: {video_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FinalAprilTagLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording_event.set()
        node.cam_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
