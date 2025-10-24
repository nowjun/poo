#!/bin/bash

echo "🔧 테스트 클라이언트 Docker 이미지 빌드 중..."
docker build -f Dockerfile.test -t test-client .

echo "🚀 테스트 클라이언트 실행 중..."
echo "서버가 실행 중인지 확인하세요 (포트 8765)"
echo ""

docker run --rm --network host test-client
