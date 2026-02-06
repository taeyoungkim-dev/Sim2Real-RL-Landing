# QAV250 Sim-to-Real: Autonomous Landing via RL

> **Deep Reinforcement Learning based Precision Landing on Moving Target**
> From Simulation (Gazebo) to Real-World (QAV250 + Pixhawk 4 + RPi 5)

## Project Overview
This project implements an autonomous landing system for a quadcopter using **Deep Reinforcement Learning (PPO)**. The agent is trained in a high-fidelity simulation environment (Gazebo) and transferred to a custom-built QAV250 drone using a Sim-to-Real pipeline.

**Key Features:**
- **Sim-to-Real:** Domain Randomization applied to bridge the reality gap (wind, sensor noise, latency).
- **Vision-based Control:** Real-time marker detection (ArUco) using a downward-facing camera.
- **Hybrid Control:** RL for XY-alignment and Rule-based control for safe descent (Z-axis).
- **Hardware Integration:** ROS 2 Humble based communication between Companion Computer (RPi 5) and Flight Controller (Pixhawk 4).

## 🛠 Hardware Specs
- **Airframe:** QAV250 Carbon Fiber Frame
- **Flight Controller:** Pixhawk 4 (running PX4 Autopilot)
- **Companion Computer:** Raspberry Pi 5 (Ubuntu 22.04 + ROS 2 Humble)
- **Sensors:** Foxeer M10Q GPS, Webcam (iriver, downward-facing)
- **Power:** 4S 1500mAh LiPo, 45A 4-in-1 ESC

## Software Stack
- **OS:** Ubuntu 22.04 LTS (WSL2 / RPi 5)
- **Middleware:** ROS 2 Humble
- **Flight Stack:** PX4 Autopilot (v1.14)
- **Simulation:** Gazebo Classic / Ignition
- **RL Framework:** PyTorch (PPO), Stable Baselines3
- **Vision:** OpenCV (ArUco Marker Detection)

## How to Run (Simulation)
```bash
# 1. Clone this repository
git clone [https://github.com/YOUR_ID/qav250_rl_landing.git](https://github.com/YOUR_ID/qav250_rl_landing.git)
cd qav250_rl_landing

# 2. Build the package
colcon build --symlink-install
source install/setup.bash

# 3. Launch Simulation (Gazebo + PX4 + RL Node)
ros2 launch qav250_landing sitl_landing.launch.py