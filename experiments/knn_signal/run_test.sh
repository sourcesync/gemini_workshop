if [[ $# -lt 1 ]]
  then
    echo "usage : ./run_test Board_IP "
    exit 1	  
fi
proc2=$(sudo sshpass -p "root" ssh -o StrictHostKeyChecking=no root@$1 'ps -o comm,pid | grep -v grep | grep gsifw ' | awk '{print $2;}' )
#echo $proc2
if [[ $proc2 == '' ]]
then
	echo "arc not running"
	exit 1
else
	echo "arc already running"
fi
PYTHON=python3.6
#PYTHON=/home/administrator/anaconda3/bin/python3
export LD_LIBRARY_PATH=00.09.00/libs
export PYTHONPATH=00.09.00:00.09.00/gnlpy:00.09.00/gnlpy/lib:00.09.00/libs
$PYTHON test4.py
echo "run_hamming completed"
