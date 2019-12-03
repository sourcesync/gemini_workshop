#!/bin/bash

set -x
set -e

#NOCACHE="--no-cache"

IMAGEBASE="geminiws_base"
TAGBASE="v1"

IMAGENB="geminiws_nb"
TAGNB="v7"

# base
docker build $NOCACHE -f Dockerfile.base -t "$IMAGEBASE:$TAGBASE" ..
docker tag "$IMAGEBASE:$TAGBASE" "us.gcr.io/gsitechnology/$IMAGEBASE:$TAGBASE"
docker push "us.gcr.io/gsitechnology/$IMAGEBASE:$TAGBASE"

# nb
docker build $NOCACHE -f Dockerfile.nb -t "$IMAGENB:$TAGNB" ..
docker tag "$IMAGENB:$TAGNB" "us.gcr.io/gsitechnology/$IMAGENB:$TAGNB"
docker push "us.gcr.io/gsitechnology/$IMAGENB:$TAGNB"

# docker hub
#docker tag "$IMAGE:$TAG" "docker.io/gosha1128/$IMAGE:$TAG"
#docker push "docker.io/gosha1128/$IMAGE:$TAG"
