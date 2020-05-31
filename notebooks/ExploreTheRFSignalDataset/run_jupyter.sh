#!/bin/bash

if [[ $# -lt 1 ]];
then
	echo "usage : ./run_jupyter.sh <PATH_TO_DATASET> [<LOCAL_IP_ADDRESS>]"
	exit 1	  
fi

if [ ! -d $1 ]; 
then
	echo "The directory '$1' does not exist."
	exit 1
fi

export PYTHONPATH="../common"
export DEEPSIGNAL_DATA_DIR="$1"

if [ -z "$2" ]; 
then
	jupyter lab
else
	jupyter lab --ip=$2
fi
