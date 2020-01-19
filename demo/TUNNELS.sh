#!/bin/bash

############################################################
# Configure for your local machine or create a data.dat file

ETH=
PASS=
MAC=
DEST=

##################################

# This dat file if present will override...
if [ -f "data.dat" ]; then
	echo "data.dat file exists."
	source data.dat
fi
#################################

echo "Using these settings:  ETH=$ETH, PASS=ssssh, MAC=$MAC"
sleep 1

#################################

#set -x

# create sub-interface as needed for 192.168.88.221 and possibly 172.17.0.1

echo "seeing if 192.168.88.221 exists locally..."
ping 192.168.88.221 -c 1 -W 1
RET="$?"
if [ ! "$RET" -eq "0" ]; then
	echo "192.168.88.221 does not exist. Adding IP to interface=$ETH..."
	if [ "$MAC" -eq "1" ]; then
		echo "creating 192 virtual IP for mac..."
		ifconfig lo0 alias 192.168.88.221
		RET="$?"
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not add IP to interface=lo0"
			exit
		fi
	else
		echo "creating virtual IP for linux..."
		sudo ifconfig "$ETH:0" 192.168.88.221
		RET="$?"
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not add IP to interface=$ETH"
			exit
		fi
	fi
else
	echo "The IP is reachable."
fi

echo "seeing if 172.17.0.1 exists locally..."
ping 172.17.0.1 -c 1 -W 1
RET="$?"
if [ ! "$RET" -eq "0" ]; then
	echo "172.17.0.1 does not exist. Adding IP to interface=$ETH..."
	if [ "$MAC" -eq "1" ]; then
		echo "creating 172 virtual IP for mac..."
		sudo ifconfig lo0 alias 172.17.0.1
		RET="$?"
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not add IP to interface=lo0"
			exit
		fi
	else
		echo "creating virtual IP for linux..."
		ifconfig "$ETH:0" 172.17.0.1
		RET="$?"
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not add IP to interface=$ETH"
			exit
		fi
	fi
else
	echo "The IP is reachable."
fi

# Let's make sure that sshpass is installed
which sshpass
RET=$?
if [ ! "$RET" -eq "0" ]; then
	echo "Please install sshpass. For ubuntu, just use apt-get.  "
	echo "For mac, use brew 'brew install https://raw.githubusercontent.com/kadwanev/bigboybrew/master/Library/Formula/sshpass.rb'"
	exit 1
fi

echo "warning: killing all current SSH tunnels on this machine!"
sudo killall ssh
sleep 1

# create ssh tunnel to apu machine ports
# TODO: removed 4999

ports=( 5000 8098 8097 8094 6379 7707 8085 8095 5432 5001 8092 7777 8096 7780 8091 8099 8093 7654 8087 )

for i in "${ports[@]}"
do
	echo "creating tunnel for $i"
	if [ "$MAC" -eq "1" ]; then
		echo "creating for mac..."
		sshpass -p $PASS ssh -L 172.17.0.1:$i:172.17.0.1:$i $DEST -fN
		RET=$?
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not create the tunnel for port=$i"
			exit 1
		fi
	else
		echo "creating for linux..."
		ssh -L 172.17.0.1:$i:172.17.0.1:$i $DEST -fN
		RET=$?
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not create the tunnel for port=$i"
			exit 1
		fi
	fi

	sleep 0.5
	echo "testing the tunnel for $i"

	tries=(1 2 3 4 5 6 7 8 9 10)
	for j in "${tries[@]}"
	do
		ps aux|grep "ssh -L"

		nc -z 172.17.0.1 $i
		RET=$?
		echo "NCRET=$RET"
		if [ ! "$RET" -eq "0" ]; then
			echo "Could not validate the tunnel for $i (try=$j). Trying again..."
		else
			break
		fi

		sleep 1
	done
	if [ ! "$RET" -eq "0" ]; then
		echo "Could not validate the tunnel for $i"
		exit 1
	fi
	echo "Tunnel validated for $i"

done

echo "creating tunnel for 8090"
if [ "$MAC" -eq "1" ]; then
	echo "creating for mac..."
	sshpass -p $PASS ssh -L 192.168.88.221:8090:172.17.0.1:8090 $DEST -fN
	RET=$?
	if [ ! "$RET" -eq "0" ]; then
		echo "Could not create the tunnel for port=$i"
		exit
	fi
else
	echo "creating for linux..."
	ssh -L 192.168.88.221:8090:172.17.0.1:8090 $DEST -fN
	if [ ! "$RET" -eq "0" ]; then
		echo "Could not create the tunnel for port=$i"
		exit
	fi
fi
	
sleep 0.1
echo "testing the tunnel"
nc -z 192.168.88.221 8090
RET=$?
if [ ! "$RET" -eq "0" ]; then
	echo "Could not validate the tunnel for $i"
	exit 1
fi

echo "Done."
