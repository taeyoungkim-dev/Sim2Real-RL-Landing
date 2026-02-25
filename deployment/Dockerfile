FROM ros:humble-ros-base

# 필수 도구 및 종속성 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 1. Micro-XRCE-DDS-Agent 설치 (통역사)
WORKDIR /root
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && cd build && \
    cmake .. && make && make install && \
    ldconfig

# 2. 작업 공간(Workspace) 생성 및 px4_msgs 설치 (단어장)
WORKDIR /ros2_ws/src
RUN git clone https://github.com/PX4/px4_msgs.git

# ROS 2 빌드
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build

# [여기에 추가!] docker exec 접속 시 자동 source를 위한 .bashrc 세팅
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# 3. 완벽한 Entrypoint 스크립트 생성 및 권한 부여
RUN echo '#!/bin/bash' > /ros_entrypoint.sh && \
    echo 'set -e' >> /ros_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /ros_entrypoint.sh && \
    echo 'source /ros2_ws/install/setup.bash' >> /ros_entrypoint.sh && \
    echo 'nohup MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600 > /tmp/agent.log 2>&1 &' >> /ros_entrypoint.sh && \
    echo 'exec "$@"' >> /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

# 4. 올바른 Entrypoint 지정
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]