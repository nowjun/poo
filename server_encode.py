
import asyncio
import websockets
import numpy as np
import torch
import torch.nn as nn
import zlib
import json
import subprocess

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
DISPLAY = ":1"
WEBSOCKET_PORT = 8765

# --- Simple CNN Encoder (Demo) ---
class Encoder(nn.Module):
    def __init__(self, original_w, original_h, c=64):
        super(Encoder, self).__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.c = c

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, c, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.conv_stack(x)

def get_latent_shape(original_w, original_h):
    # Calculate the output shape after the convolutional layers
    w = original_w
    h = original_h
    for _ in range(4): # 4 convolutional layers with stride 2
        w = (w - 5 + 2 * 2) // 2 + 1
        h = (h - 5 + 2 * 2) // 2 + 1
    return w, h

# --- Main Server Logic ---
async def server(websocket, path):
    print("Client connected.")

    # Initialize the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Encoder(WIDTH, HEIGHT).to(device)
    model.eval()

    latent_w, latent_h = get_latent_shape(WIDTH, HEIGHT)

    # Start ffmpeg to capture the screen
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'x11grab',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-i', DISPLAY,
        '-vf', 'format=rgb24',
        '-f', 'rawvideo',
        'pipe:1'
    ]

    proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        while True:
            # Read a frame from ffmpeg's stdout
            frame_bytes = await proc.stdout.read(WIDTH * HEIGHT * 3)
            if not frame_bytes:
                print("FFmpeg process stopped.")
                break

            # Convert frame to tensor
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            # Encode the frame
            with torch.no_grad():
                latent = model(frame_tensor)

            # Quantize and compress
            scale = latent.abs().max() / 127.0
            quantized_latent = (latent / scale).clamp(-128, 127).to(torch.int8)
            compressed_latent = zlib.compress(quantized_latent.cpu().numpy().tobytes())

            # Prepare header
            header = {
                "w": latent_w,
                "h": latent_h,
                "c": model.c,
                "scale": scale.item(),
                "model_ver": 1,
                "orig_w": WIDTH,
                "orig_h": HEIGHT,
                "timestamp": asyncio.get_event_loop().time()
            }
            header_bytes = json.dumps(header).encode('utf-8')
            header_len_bytes = len(header_bytes).to_bytes(4, 'big')

            # Send data
            await websocket.send(header_len_bytes + header_bytes + compressed_latent)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        proc.terminate()
        await proc.wait()


if __name__ == "__main__":
    start_server = websockets.serve(server, "0.0.0.0", WEBSOCKET_PORT)

    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"WebSocket server started on port {WEBSOCKET_PORT}.")
    asyncio.get_event_loop().run_forever()
