#!/bin/bash

openssl rand -hex 32

read -p "Did you set the secretToken in config.yaml (y/n)? " answer
case ${answer:0:1} in
    y|Y )
        echo Yes
    ;;
    * )
        exit 1
    ;;
esac

helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
helm repo update

RELEASE=jhub
NAMESPACE=jhub

helm upgrade --install $RELEASE jupyterhub/jupyterhub \
  --namespace $NAMESPACE  \
  --version=0.8.2 \
  --values config.yaml

kubectl get pod --namespace jhub

kubectl config set-context $(kubectl config current-context) --namespace ${NAMESPACE:-jhub}

#kubectl get service --namespace jhub

#kubectl describe service proxy-public --namespace jhub


