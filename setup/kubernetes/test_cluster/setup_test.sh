#!/bin/bash

set -x
set -e

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


./setup_test_cluster.sh

sleep 1

./setup_test_helm.sh

sleep 1

./setup_test_jupyterhub.sh passthru

while True; do

	kubectl get pods

	kubectl get service --namespace test

	sleep 1
done
