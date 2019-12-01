#!/bin/bash
set -x
set -e
#docker cp 040398d0ea43:/opt/conda/lib/python3.6/site-packages/faiss-1.6.0-py3.6.egg ./

docker cp 040398d0ea43:/opt/faiss ./

docker cp 040398d0ea43:/usr/lib/x86_64-linux-gnu/libopenblas.so.0 ./
docker cp 040398d0ea43:/usr/lib/x86_64-linux-gnu/liblapack.so.3 ./
