#!/bin/bash
# Full Chitrak + Isaac Lab setup from scratch.
# Run once after cloning this repo on a fresh server.
#
# Usage:
#   git clone https://github.com/YOUR_USERNAME/chitrak-isaaclab-setup.git
#   cd chitrak-isaaclab-setup
#   bash setup.sh

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDIO="/teamspace/studios/this_studio"

echo "=== [1/4] Cloning Isaac Lab v2.3.2 ==="
if [ ! -d "$STUDIO/IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$STUDIO/IsaacLab"
    cd "$STUDIO/IsaacLab" && git checkout v2.3.2
else
    echo "IsaacLab already exists, skipping."
fi

echo "=== [2/4] Copying integration files ==="
mkdir -p "$STUDIO/IsaacLab/source/chitrak_integration"
cp -r "$REPO_DIR/chitrak_integration/"* "$STUDIO/IsaacLab/source/chitrak_integration/"

echo "=== [3/4] Patching docker-compose.yaml ==="
COMPOSE="$STUDIO/IsaacLab/docker/docker-compose.yaml"
PATCH='    # Chitrak robot meshes\n  - type: bind\n    source: /teamspace/studios/this_studio/chitrak-rl\n    target: /workspace/chitrak-rl'

if grep -q "chitrak-rl" "$COMPOSE"; then
    echo "docker-compose.yaml already patched, skipping."
else
    sed -i "s|    # This volume is used to store the history of the bash shell|${PATCH}\n    # This volume is used to store the history of the bash shell|" "$COMPOSE"
    echo "Patched docker-compose.yaml."
fi

echo "=== [4/4] Starting Isaac Lab container ==="
echo "Starting container (answer N to X11 prompt)..."
cd "$STUDIO/IsaacLab/docker"
echo "N" | ./container.py start

echo ""
echo "====================================="
echo " Setup complete!"
echo "====================================="
echo ""
echo "Enter the container:"
echo "  cd $STUDIO/IsaacLab/docker && ./container.py enter"
echo ""
echo "Then inside the container:"
echo "  export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:\$PYTHONPATH"
echo "  ./isaaclab.sh -p -m pip install flatdict"
echo ""
echo "If USD not yet generated, run:"
echo "  ./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/convert_urdf_to_usd.py"
echo ""
echo "Verify robot:"
echo "  ./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/verify_robot.py"
