#!/bin/bash

##################################
# Configure for your local machine

ETH=ppp0
PASS=george

##################################

#set -x

# create sub-interface as needed for 192.168.88.221

echo "seeing if 192.168.88.221 exists locally..."
ping 192.168.88.221 -c 1 -W 1
RET="$?"
if [ ! "$RET" -eq "0" ]; then
	echo "192.168.88.221 does not exist. Adding IP to interface=$ETH..."
	ifconfig "$ETH:0" 192.168.88.221
	RET="$?"
	if [ ! "$RET" -eq "0" ]; then
		echo "Could not add IP to interface=$ETH"
		exit
	fi
else
	echo "The IP is reachable."
fi

echo "warning: killing all current SSH tunnels on this machine!"
killall ssh
sleep 1

# create ssh tunnel to apu machine ports

ports=(4999 5000 7654 8093 8098 8097 8094 8091 6379 7707 8087 8085 8095 8099 5432 5001 8092 7777 8096 7780 )

for i in "${ports[@]}"
do
	echo "creating tunnel for $i"
	sshpass -p "$PASS" ssh -L 172.17.0.1:$i:172.17.0.1:$i george@192.168.99.21 -fN
	RET=$?
	if [ ! "$RET" -eq "0" ]; then
		echo "Could not create the tunnel for port=$i"
		exit
	fi
done

echo "creating tunnel for 8090"
sshpass -p "$PASS" ssh -L 192.168.88.221:8090:172.17.0.1:8090 george@192.168.99.21 -fN
if [ ! "$RET" -eq "0" ]; then
	echo "Could not create the tunnel for port=$i"
	exit
fi

echo "Done."
