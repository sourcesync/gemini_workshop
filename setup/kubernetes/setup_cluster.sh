#!/bin/bash

set -x
set -e

gcloud components install kubectl

gcloud container clusters create \
  --machine-type n1-standard-2 \
  --num-nodes 7 \
  --zone us-central1-b \
  --cluster-version latest \
  geminiws

kubectl get node

kubectl create clusterrolebinding cluster-admin-binding \
  --clusterrole=cluster-admin \
  --user=george.williams@gmail.com

#gcloud beta container node-pools create user-pool \
#  --machine-type n1-standard-2 \
#  --num-nodes 0 \
#  --enable-autoscaling \
#  --min-nodes 0 \
#  --max-nodes 7 \
#  --node-labels hub.jupyter.org/node-purpose=user \
#  --node-taints hub.jupyter.org_dedicated=user:NoSchedule \
#  --zone us-central1-b \
#  --cluster geminiws

kubectl get node
