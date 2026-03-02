import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

class Figure8Node(Node):
    def __init__(self):
        super().__init__('figure_8_node')
        #offboard publisher
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',12)
        #trajectory publisher
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint',10)
        #heartbeat publisher
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command',10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.timer_count == 20:
            self.get_logger().info("Offboard topic is published")
        
        elif self.timer_count == 30:
            self.get_logger().info("Arming topic is published")
        #TODO : When will be end
        self.timer_count += 1
    
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)

