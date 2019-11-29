#!/bin/bash

RELEASE=jhub

helm upgrade $RELEASE jupyterhub/jupyterhub   --version=0.8.2   --values config.yaml


