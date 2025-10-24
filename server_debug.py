#!/usr/bin/env python3

import asyncio
import websockets
import numpy as np
import torch
import torch.nn as nn
import zlib
import json
import subprocess
import time

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
    w = original_w
    h = original_h
    for _ in range(4):
        w = (w - 5 + 2 * 2) // 2 + 1
        h = (h - 5 + 2 * 2) // 2 + 1
    return w, h

# --- Main Server Logic ---
async def server(websocket, path):
    print("✅ 클라이언트 연결됨")

    # Initialize the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 사용 중인 디바이스: {device}")
    model = Encoder(WIDTH, HEIGHT).to(device)
    model.eval()

    latent_w, latent_h = get_latent_shape(WIDTH, HEIGHT)
    print(f"📐 Latent 크기: {latent_w}x{latent_h}x{model.c}")

    # Test frame generation instead of screen capture
    print("🎬 테스트 프레임 생성 시작...")
    
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Generate test pattern instead of screen capture
            test_frame = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(test_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

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
                "timestamp": time.time()
            }
            header_bytes = json.dumps(header).encode('utf-8')
            header_len_bytes = len(header_bytes).to_bytes(4, 'big')

            # Send data
            await websocket.send(header_len_bytes + header_bytes + compressed_latent)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 프레임 #{frame_count} | FPS: {fps:.1f}")

            # 30 FPS로 제한
            await asyncio.sleep(1/30)

    except websockets.exceptions.ConnectionClosed:
        print("❌ 클라이언트 연결 끊어짐")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    print(f"🚀 WebSocket 서버 시작 중... 포트: {WEBSOCKET_PORT}")
    start_server = websockets.serve(server, "0.0.0.0", WEBSOCKET_PORT)
    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"✅ 서버가 포트 {WEBSOCKET_PORT}에서 실행 중입니다.")
    asyncio.get_event_loop().run_forever()
