#!/bin/bash

set -x
set -x

RELEASE=jhub

helm upgrade $RELEASE jupyterhub/jupyterhub   --debug --version=0.8.2   --values config.yaml

#./delete_all_pods.sh

while True; do

        kubectl get pods

        kubectl get service --namespace jhub

        sleep 1
done
