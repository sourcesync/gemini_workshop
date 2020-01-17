#!/bin/bash

echo "5000"
sudo ssh -L 172.17.0.1:5000:172.17.0.1:5000 george@192.168.99.21 -fN
echo "ret=$?"

echo "7654"
sudo ssh -L 172.17.0.1:7654:172.17.0.1:7654 george@192.168.99.21 -fN
echo "ret=$?"

echo "8093"
sudo ssh -L 172.17.0.1:8093:172.17.0.1:8093 george@192.168.99.21 -fN
echo "ret=$?"

echo "8098"
sudo ssh -L 172.17.0.1:8098:172.17.0.1:8098 george@192.168.99.21 -fN
echo "ret=$?"

echo "8097"
sudo ssh -L 172.17.0.1:8097:172.17.0.1:8097 george@192.168.99.21 -fN
echo "ret=$?"

echo "8094"
sudo ssh -L 172.17.0.1:8094:172.17.0.1:8094 george@192.168.99.21 -fN
echo "ret=$?"

echo "8091"
sudo ssh -L 172.17.0.1:8091:172.17.0.1:8091 george@192.168.99.21 -fN
echo "ret=$?"

echo "6379"
sudo ssh -L 172.17.0.1:6379:172.17.0.1:6379 george@192.168.99.21 -fN
echo "ret=$?"

echo "7707"
sudo ssh -L 172.17.0.1:7707:172.17.0.1:7707 george@192.168.99.21 -fN
echo "ret=$?"

echo "8087"
sudo ssh -L 172.17.0.1:8087:172.17.0.1:8087 george@192.168.99.21 -fN
echo "ret=$?"

echo "8085"
sudo ssh -L 172.17.0.1:8085:172.17.0.1:8085 george@192.168.99.21 -fN
echo "ret=$?"

echo "8095"
sudo ssh -L 172.17.0.1:8095:172.17.0.1:8095 george@192.168.99.21 -fN
echo "ret=$?"

echo "8099"
sudo ssh -L 172.17.0.1:8099:172.17.0.1:8099 george@192.168.99.21 -fN
echo "ret=$?"

echo "5432"
sudo ssh -L 172.17.0.1:5432:172.17.0.1:5432 george@192.168.99.21 -fN
echo "ret=$?"

echo "5001"
sudo ssh -L 172.17.0.1:5001:172.17.0.1:5001 george@192.168.99.21 -fN
echo "ret=$?"

echo "8092"
sudo ssh -L 172.17.0.1:8092:172.17.0.1:8092 george@192.168.99.21 -fN
echo "ret=$?"

echo "7777"
sudo ssh -L 172.17.0.1:7777:172.17.0.1:7777 george@192.168.99.21 -fN
echo "ret=$?"

echo "8096"
sudo ssh -L 172.17.0.1:8096:172.17.0.1:8096 george@192.168.99.21 -fN
echo "ret=$?"

echo "7780"
sudo ssh -L 172.17.0.1:7780:172.17.0.1:7780 george@192.168.99.21 -fN
echo "ret=$?"

echo "8090"
sudo ssh -L 192.168.88.221:8090:172.17.0.1:8090 george@192.168.99.21 -fN


