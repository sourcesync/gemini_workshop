#!/bin/bash

docker build -f Dockerfile -t geminiws/test:v1 ..

docker tag geminiws/test:v1 us.gcr.io/gsitechnology/geminiws/test:v1
docker push us.gcr.io/gsitechnology/geminiws/test:v1
