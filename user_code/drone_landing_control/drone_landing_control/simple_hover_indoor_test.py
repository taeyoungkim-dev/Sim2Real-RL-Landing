import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, VehicleAttitudeSetpoint, VehicleCommand

class SimpleAttitudeNode(Node):
    def __init__(self):
        super().__init__('simple_attitude_node')

        # [수정됨] TrajectorySetpoint 대신 VehicleAttitudeSetpoint를 퍼블리시합니다.
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.attitude_setpoint_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_attitude_setpoint()

        if self.timer_count == 20:  # 2.0초: 오프보드 모드만 먼저 진입!
            self.get_logger().info("1단계: Offboard 모드 진입 요청")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        elif self.timer_count == 30:  # 3.0초 (1초 뒤): 시동(Arm) 명령 발사!
            self.get_logger().info("2단계: 1초 대기 후 시동(Arm) 명령 발사!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        elif self.timer_count == 120:  # 12초 뒤 착륙(시동 끄기)
            self.get_logger().info("🛑 테스트 종료. 시동을 끕니다(Disarm)!")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0) # 강제 Disarm

        self.timer_count += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        # [핵심] position을 끄고, attitude(기울기) 제어만 켭니다!
        msg.position = False  
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True   
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_attitude_setpoint(self):
        msg = VehicleAttitudeSetpoint()
        # 쿼터니언 [w, x, y, z] : [1.0, 0.0, 0.0, 0.0]은 완벽한 수평(Roll=0, Pitch=0, Yaw=0)을 의미합니다.
        msg.q_d = [1.0, 0.0, 0.0, 0.0]
        # [핵심] Z축 파워(Thrust): NED 좌표계이므로 -0.1은 위쪽으로 10%의 파워를 주라는 뜻입니다.
        msg.thrust_body = [0.0, 0.0, -0.1] 
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.attitude_setpoint_pub.publish(msg)

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
    node = SimpleAttitudeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()