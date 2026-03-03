import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
import math

class Figure8Node(Node):
    def __init__(self):
        super().__init__('figure8_node')

        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.timer_count = 0
        
        # 8자 비행을 위한 시간 변수 및 파라미터
        self.flight_time = 0.0
        self.radius_x = 4.0  # X축 반경 (미터)
        self.radius_y = 4.0  # Y축 반경 (미터)
        self.omega = 0.8     # 비행 속도 (값이 커질수록 빨라짐)
        self.flight_altitude = -2.0 # 비행 고도 2m

    def timer_callback(self):
        self.publish_offboard_control_mode()

        # --- 상태 머신 (시간 흐름에 따른 동작 제어) ---
        if self.timer_count == 20:
            self.get_logger().info("1단계: Offboard 모드 진입")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        elif self.timer_count == 30:
            self.get_logger().info("2단계: 시동(Arm) 및 안전 이륙 대기")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        elif self.timer_count < 80:
            # 3~8초: 제자리에서 2m까지 부드럽게 이륙
            self.publish_hover_setpoint()

        elif self.timer_count < 300:
            # 8~30초: 본격적인 고속 8자 비행 시작!
            if self.timer_count == 80:
                self.get_logger().info("3단계: 고속 8자 기동 시작! 🚀")
            
            self.publish_figure8_setpoint()
            self.flight_time += self.timer_period  # 궤적 계산을 위한 시간 증가

        elif self.timer_count == 300:
            self.get_logger().info("4단계: 중심 복귀 및 자동 착륙 시작 🛑")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True  
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_hover_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [0.0, 0.0, self.flight_altitude]
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_figure8_setpoint(self):
        msg = TrajectorySetpoint()
        
        # [수학적 궤적 계산]
        # X, Y 위치 계산 (Lemniscate 궤적)
        x = self.radius_x * math.sin(self.omega * self.flight_time)
        y = self.radius_y * math.sin(2.0 * self.omega * self.flight_time) / 2.0
        
        # 위치 미분을 통한 X, Y 속도 계산 (Feedforward 용도)
        vx = self.radius_x * self.omega * math.cos(self.omega * self.flight_time)
        vy = self.radius_y * self.omega * math.cos(2.0 * self.omega * self.flight_time)
        
        # 기수(Yaw)가 비행 진행 방향을 바라보도록 각도 계산
        yaw_angle = math.atan2(vy, vx)

        # Setpoint 입력
        msg.position = [x, y, self.flight_altitude]
        msg.velocity = [vx, vy, 0.0]
        msg.yaw = yaw_angle
        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Figure8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()