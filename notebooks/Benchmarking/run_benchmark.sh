#!/bin/bash

if [[ $# -lt 2 ]]
  then
    echo "usage : ./run_benchmark Board_IP /path/to/data"
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
export LD_LIBRARY_PATH=/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/libs
export PYTHONPATH=/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/gnlpy:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/gnlpy/lib:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/libs
$PYTHON benchmark.py $2
echo "run_benchmark completed"
