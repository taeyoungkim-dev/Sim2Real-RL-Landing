import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import time
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

#pub,sub msg
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus, VehicleOdometry

#Env가 ros2 node를 겸함
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

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

        #Gym space
        self.action_space = spaces.Box(low=1.0,high=1.0,shape=(3,),dtype=np.float32)

        #Observation
        #Drone velocity = 3
        #Drone roll,pitch,yaw = 3
        #Drone angular velocity = 3
        #Marker relative pos = 3
        #Marker velocity = 3
        #Formar action = 3
        # Sumation of field size = 18
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        #Sampling time
        self.dt = 0.1
    
    #Callback function
    def odom_cb(self,msg):
        self.current_pos = np.array([msg.positive[0],msg.position[1],msg.position[2]])
        self.current_vel = np.array([msg.velocity[0],msg.velocity[1],msg.velocity[2]])

    #reset funciton
    def reset(self,seed=None, option = None):
        super().reset(seed=seed)

        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)

        # Arming
        armed = False
        for i in range(50):
            self._publish_velocity([0.0, 0.0, 0.0])
            self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if self.nav_state == 14 and self.arming_state == 2:
                armed = True
                break
        
        if not armed:
            print(">>> Arming Failed. Retrying...")
            return self.reset(seed=seed) # 재귀 호출로 다시 시도