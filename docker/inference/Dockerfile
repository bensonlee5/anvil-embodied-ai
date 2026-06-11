# Use ROS2 Jazzy as base
FROM ros:jazzy-ros-base

ARG LEROBOT_VERSION=0.5.0
ARG LEROBOT_EXTRAS=""

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Base lerobot install — cached as long as version doesn't change
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed \
    numpy packaging "lerobot==${LEROBOT_VERSION}" matplotlib && \
    pip3 install --break-system-packages --no-cache-dir "setuptools<80"

# Extras — separate layer, only rebuilt when LEROBOT_EXTRAS changes
RUN if [ -n "${LEROBOT_EXTRAS}" ]; then \
        pip3 install --break-system-packages --no-cache-dir \
            "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"; \
    fi

# Patch lerobot v0.5.0 bug: groot_n1.py has a malformed @dataclass that crashes
# on import. Remove groot lines from policies/__init__.py since we don't use GR00T.
RUN find /usr/local/lib -path "*/lerobot/policies/__init__.py" -exec sed -i '/groot/d' {} \;

# Copy anvil_shared (pure Python, no ROS build needed — must be before lerobot_control)
COPY packages/anvil_shared/src /workspace/anvil_shared_src

# Copy ROS2 package (changes often → late layer)
# anvil_msgs must be built before lerobot_control (it depends on it for EE mode)
COPY ros2/src/anvil_msgs /workspace/src/anvil_msgs
COPY ros2/src/lerobot_control /workspace/src/lerobot_control

# Copy configs (lerobot_control + cyclonedds)
COPY configs /workspace/configs

# Make anvil_shared importable everywhere in the container
ENV PYTHONPATH="/workspace/anvil_shared_src:${PYTHONPATH:-}"

# Build ROS2 workspace
RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

# Set entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
