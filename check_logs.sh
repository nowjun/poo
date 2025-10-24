#!/bin/bash

echo "🔍 실행 중인 컨테이너 확인..."
docker ps

echo ""
echo "📊 서버 로그 확인 (최근 50줄)..."
SERVER_CONTAINER=$(docker ps -q --filter ancestor=remote-desktop-encoder)
if [ ! -z "$SERVER_CONTAINER" ]; then
    echo "서버 컨테이너 ID: $SERVER_CONTAINER"
    docker logs --tail 50 $SERVER_CONTAINER
else
    echo "❌ 서버 컨테이너가 실행 중이 아닙니다."
fi

echo ""
echo "📊 테스트 클라이언트 로그 확인 (최근 50줄)..."
CLIENT_CONTAINER=$(docker ps -q --filter ancestor=test-client)
if [ ! -z "$CLIENT_CONTAINER" ]; then
    echo "클라이언트 컨테이너 ID: $CLIENT_CONTAINER"
    docker logs --tail 50 $CLIENT_CONTAINER
else
    echo "❌ 테스트 클라이언트 컨테이너가 실행 중이 아닙니다."
fi
