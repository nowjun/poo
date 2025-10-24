#!/bin/bash

# Build the Docker image
docker build -t remote-desktop-encoder .

# Run the Docker container
docker run --gpus all -p 8765:8765 --rm remote-desktop-encoder
