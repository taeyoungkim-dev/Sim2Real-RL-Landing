import numpy as np
import rclpy
import time
from rclpy.node import Node
from px4_msgs.msg import TrajectorySetpoint, VehicleCommand, OffboardControlMode, VehicleOdometry, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RuleBasedLandingNode(Node):
    def __init__(self):
        super().__init__('rule_based_landing_node')

        #Init ros2
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('gym_env_node')

        #QoS
        qos_profile = QoSProfile(
            Reliability = ReliabilityPolicy.BEST_EFFORT,
            History = HistoryPolicy.KEEP_LAST,
            depth = 1 
        )

        #Publisher
        self.pub_offboard_mode = self.node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.pub_trajectory = self.node.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.pub_vehicle_command = self.node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        #Subscriber
        self.sub_odom = self.node.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_profile)
        self.sub_status = self.node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, qos_profile)
        
        #Variable
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.target_pos = np.zeros(3)

        #Callback function
        def odom_cb(self,msg):
            self.current_pos = np.array([msg.positive[0],msg.position[1],msg.position[2]])
            self.current_vel = np.array([msg.velocity[0],msg.velocity[1],msg.velocity[2]])