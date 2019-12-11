if [[ $# -lt 2 ]]
  then
    echo "usage : ./run_hamming iter_num (0 for infinite) Board_IP "
    exit 1	  
fi
proc2=$(sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$2 'ps -o comm,pid | grep -v grep | grep gsifw ' | awk '{print $2;}' )
#echo $proc2
if [[ $proc2 == '' ]]
then
	echo "run arc"
	nohup sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$2 '/run/media/mmcblk0p1/system/bin/run_app' &
	echo "Waiting for the arc to load..."
	sleep 10
else
	echo "arc already running"
fi
export LD_LIBRARY_PATH=00.09.00/libs; export PYTHONPATH=00.09.00:00.09.00/gnlpy:00.09.00/gnlpy/lib:00.09.00/libs; python3.6 knn_binding_usa.py $1
proc=$(sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$2 'ps -o comm,pid | grep -v grep | grep gsifw ' | awk '{print $2;}' )
if [[ $proc != '' ]]
then
	echo "terminate arc"
	sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$2 "kill -9 $proc"
fi
echo "run_hamming completed"
