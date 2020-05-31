#!/bin/bash

if [[ $# -lt 1 ]]
  then
    echo "usage : ./run_jupyter <PATH_TO_DATASET> [ <LOCAL_IP_ADDRESS> ]"
    exit 1	  
fi

if [ ! -d $1 ];
then
	echo "The directory '$1' does not exist."
	exit 1
fi

export DEEPSIGNAL_DATA_DIR="$1"

export LD_LIBRARY_PATH=/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/libs

export PYTHONPATH=../common:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/gnlpy:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/gnlpy/lib:/opt/gemini_eval_100.9.10.2-rc/gemini_release/00.17.00/libs

if [ -z "$2" ];
then
	jupyter lab
else
	jupyter lab --ip=$2
fi
