#!/bin/bash
# Creates the persistent "univla" container (first run only) and then enters it.
# Based on the lab's launch_persisting_container.sh.
#
# Usage:   bash docker/launch_container.sh
# Exit:    type `exit` (the container stops, but is NOT deleted)
# Delete:  docker rm univla   (code and data are safe: they live on the host)

container_name="univla"
image="univla:cu121"

code_dir="$HOME/projects/univla"   # -> /workspace/univla
data_dir="$HOME/data/univla"       # -> /data

if ! docker container inspect "$container_name" > /dev/null 2>&1; then
    echo "[launch] Creating container '$container_name' from image '$image'..."
    docker create --gpus all -it \
        --net=host \
        --shm-size=16g \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --mount type=bind,source="$code_dir",target=/workspace/univla \
        --mount type=bind,source="$data_dir",target=/data \
        --name "$container_name" \
        "$image" \
        bash
fi

echo "[launch] Entering container '$container_name'..."
docker start -i "$container_name"
