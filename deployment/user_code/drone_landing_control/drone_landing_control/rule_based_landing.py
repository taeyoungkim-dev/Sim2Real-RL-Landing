import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

class RuleBasedLandingNode(Node):
    def __init__(self):
        super.__init__('rule_based_landing_node')
        
        self.offboard_ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.timer_count = 0
    
    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()