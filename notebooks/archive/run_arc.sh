if [[ $# -lt 1 ]]
  then
    echo "usage : ./run_arc.sh Board_IP "
    exit 1	  
fi
proc2=$(sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$1 'ps -o comm,pid | grep -v grep | grep gsifw ' | awk '{print $2;}' )
#echo $proc2
if [[ $proc2 == '' ]]
then
	echo "run arc"
	nohup sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$1 '/run/media/mmcblk0p1/system/bin/run_app' &
	echo "Waiting for the arc to load..."
	sleep 10
else
	echo "arc already running"
fi
