#!/bin/bash

export HOSTNAME=172.17.0.1
export REDIS_VOLUME_DIR=/var/lib/redis/volume
export POSTGRESQL_VOLUME_DIR=/var/lib/postgresql/volume

docker-compose -f docker-compose-apu-12.yml up -d --scale python-face-recognition=0 --scale python-faiss-float-plus=0 --scale python-faiss-benchmark-api=0


