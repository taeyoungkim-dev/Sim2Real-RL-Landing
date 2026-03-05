import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

TARGET_ALTITUE = 10.0
class OutdoorHoverNode(Node):
    def __init__(self):
        super().__init__('outdoor_hover_node')

        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0
        
        # [핵심 수정 1] 목표 고도를 실시간으로 변경하기 위한 변수 생성 (초기값: 지면)
        self.target_altitude = 0.0 

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.timer_count == 20:  # 2.0초: 모드 변경
            self.get_logger().info("1단계: 야외 비행 Offboard 모드 진입!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        elif self.timer_count == 30:  # 3.0초: 시동 걸기 (이때 목표 고도는 여전히 0.0m)
            self.get_logger().info("2단계: 시동(Arm) 완료. 이륙 대기 🚀")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        # [핵심 수정 2] 시동 후 2초 대기 후 이륙 지령 하달
        elif self.timer_count == 50:  # 5.0초: 1.5m 이륙 시작
            self.get_logger().info("3단계: 고도 1.5m로 이륙(Takeoff) 시작!")
            self.target_altitude = -1*TARGET_ALTITUE

        elif self.timer_count == 150:  # 15.0초: 10초간 체공 후 자동 착륙
            self.get_logger().info("4단계: 비행 완료. 자동 착륙(Land) 시작 🛑")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        # [핵심 수정 3] 고정값(-1.5) 대신 상태 변수를 사용하여 이륙 시점을 제어
        msg.position = [0.0, 0.0, self.target_altitude] 
        msg.yaw = 0.0
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
    node = OutdoorHoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()