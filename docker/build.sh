#!/bin/bash

set -x

IMAGE="geminiws"
TAG="v6"

docker build -f Dockerfile -t "$IMAGE:$TAG" ..

docker tag "$IMAGE:$TAG" "us.gcr.io/gsitechnology/$IMAGE:$TAG"
docker push "us.gcr.io/gsitechnology/$IMAGE:$TAG"

docker tag "$IMAGE:$TAG" "docker.io/gosha1128/$IMAGE:$TAG"
docker push "docker.io/gosha1128/$IMAGE:$TAG"
