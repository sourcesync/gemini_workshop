#!/bin/bash

RELEASE=jhub

helm upgrade $RELEASE jupyterhub/jupyterhub   --debug --version=0.8.2   --values config.yaml

kubectl get service --namespace jhub

