#!/bin/bash

set -x

kubectl delete --all pods --namespace=jhub

helm delete jhub

kubectl delete namespace jhub

gcloud container clusters list

gcloud container clusters delete geminiws --zone us-central1-b

