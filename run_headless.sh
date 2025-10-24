#!/bin/bash

# Build the Docker image
docker build -t remote-desktop-encoder .

# Run the Docker container in headless mode
docker run --gpus all -p 8765:8765 --rm \
  -e DISPLAY=:1 \
  -e QT_X11_NO_MITSHM=1 \
  -e _X11_NO_MITSHM=1 \
  -e _MITSHM=0 \
  remote-desktop-encoder
