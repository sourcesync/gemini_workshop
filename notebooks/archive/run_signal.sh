if [[ $# -lt 2 ]]
  then
    echo "usage : ./run_hamming iter_num (0 for infinite) Board_IP "
    exit 1	  
fi
proc2=$(sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$2 'ps -o comm,pid | grep -v grep | grep gsifw ' | awk '{print $2;}' )
#echo $proc2
if [[ $proc2 == '' ]]
then
	echo "arc not running"
	exit 1
else
	echo "arc already running"
fi
which python3.6
#export LD_LIBRARY_PATH=00.09.00/libs; export PYTHONPATH=00.09.00:00.09.00/gnlpy:00.09.00/gnlpy/lib:00.09.00/libs; python3.6 knn_signal.py $1
export LD_LIBRARY_PATH=00.15.00/libs; export PYTHONPATH=00.15.00:00.15.00/gnlpy:00.15.00/gnlpy/lib:00.15.00/libs; python3.6 knn_signal.py $1
echo "run_hamming completed"
