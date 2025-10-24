#!/usr/bin/env python3

import asyncio
import websockets
import json
import zlib
import time
import numpy as np
import torch
import torch.nn as nn

# Import the Decoder class from the model file
from decoder_model import Decoder

SERVER_URI = "ws://localhost:8765"

async def test_client():
    try:
        # Initialize Decoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 사용 중인 디바이스: {device}")
        
        # Load decoder model
        model = Decoder(c=64).to(device)
        model.eval()
        print("📦 디코더 모델 로드 완료")
        
        async with websockets.connect(SERVER_URI, max_size=1_000_000) as websocket:
            print(f"✅ 서버에 연결됨: {SERVER_URI}")
            
            frame_count = 0
            start_time = time.time()
            decode_times = []
            
            while True:
                try:
                    # 메시지 수신
                    message = await websocket.recv()
                    
                    # 헤더 파싱
                    header_len = int.from_bytes(message[:4], 'big')
                    header_json = message[4:4 + header_len].decode('utf-8')
                    header = json.loads(header_json)
                    payload = message[4 + header_len:]
                    
                    # 압축 해제
                    decompressed = zlib.decompress(payload)
                    
                    # 디코딩 시작 시간 측정
                    decode_start = time.time()
                    
                    # Convert to tensor and decode
                    latent_int8 = np.frombuffer(decompressed, dtype=np.int8)
                    latent_float32 = latent_int8.astype(np.float32) * header['scale']
                    latent_tensor = torch.from_numpy(latent_float32).reshape(1, header['c'], header['h'], header['w']).to(device)
                    
                    # Decode the frame
                    with torch.no_grad():
                        output_tensor = model(latent_tensor)
                    
                    # 디코딩 완료 시간 측정
                    decode_end = time.time()
                    decode_time = (decode_end - decode_start) * 1000  # ms
                    decode_times.append(decode_time)
                    
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # 지연시간 계산
                    latency = (current_time - header['timestamp']) * 1000
                    
                    # 평균 디코딩 시간 계산
                    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
                    
                    print(f"📊 프레임 #{frame_count} | FPS: {fps:.1f} | 지연: {latency:.1f}ms | 크기: {len(payload)} bytes | 디코딩: {decode_time:.1f}ms (평균: {avg_decode_time:.1f}ms)")
                    
                    # 10초마다 통계 출력
                    if frame_count % 100 == 0:
                        avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
                        print(f"📈 총 {frame_count}개 프레임 수신 완료 (평균 FPS: {fps:.1f}, 평균 디코딩: {avg_decode_time:.1f}ms)")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("❌ 서버 연결이 끊어졌습니다.")
                    break
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    break
                    
    except ConnectionRefusedError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 연결 오류: {e}")

if __name__ == "__main__":
    print("🔍 서버 연결 테스트 시작...")
    try:
        asyncio.run(test_client())
    except KeyboardInterrupt:
        print("\n⏹️ 테스트 중단됨")
