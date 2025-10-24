
import asyncio
import websockets
import numpy as np
import torch
import zlib
import json
import cv2
import time

# Import the Decoder class from the model file
from decoder_model import Decoder

# --- Configuration ---
SERVER_URI = "ws://localhost:8765"

# --- Main Client Logic ---
async def client():
    # --- 1. Initialize Decoder ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # In a real application, you would load a trained model.
    # For this demo, we instantiate a new decoder.
    # e.g., model.load_state_dict(torch.load('decoder_weights.pth'))
    model = Decoder(c=64).to(device)
    model.eval()
    print("Decoder model loaded.")

    # --- 2. Performance Tracking ---
    frame_count = 0
    last_fps_time = time.time()
    fps = 0

    # --- 3. Connect to Server ---
    async with websockets.connect(SERVER_URI, max_size=1_000_000) as websocket:
        print(f"Connected to server at {SERVER_URI}")
        while True:
            # --- 4. Receive and Parse Data ---
            message = await websocket.recv()
            header_len = int.from_bytes(message[:4], 'big')
            header_json = message[4:4 + header_len].decode('utf-8')
            header = json.loads(header_json)
            payload = message[4 + header_len:]

            # --- 5. Decompress and Decode ---
            decompressed = zlib.decompress(payload)
            
            latent_int8 = np.frombuffer(decompressed, dtype=np.int8)
            latent_float32 = latent_int8.astype(np.float32) * header['scale']
            
            latent_tensor = torch.from_numpy(latent_float32).reshape(1, header['c'], header['h'], header['w']).to(device)

            with torch.no_grad():
                output_tensor = model(latent_tensor)

            # --- 6. Render to OpenCV Window ---
            img_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Convert from RGB (PyTorch) to BGR (OpenCV)
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # --- 7. Calculate and Display Stats ---
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = now

            latency_ms = (now - header['timestamp']) * 1000

            cv2.putText(img_bgr, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Latency: {latency_ms:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Remote Desktop", img_bgr)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(client())
    except KeyboardInterrupt:
        print("Client stopped by user.")
