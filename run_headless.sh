#!/bin/bash

# Build the Docker image using Dockerfile.server
docker build -f Dockerfile.server -t remote-desktop-encoder .

# Run the Docker container with appuser
docker run --gpus all -p 8765:8765 --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=:1 \
  -e QT_X11_NO_MITSHM=1 \
  -e _X11_NO_MITSHM=1 \
  -e _MITSHM=0 \
  remote-desktop-encoder
