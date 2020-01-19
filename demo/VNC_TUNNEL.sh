#!/bin/bash

# check root
if [[ $EUID -eq 0 ]]; then
   echo "This script does not need to be run as root." 
   exit 1
fi

DEST=
if [ -f "data.dat" ]; then
	echo "Found data.dat..."
	source data.dat
fi

echo "Using settings: DEST=$DEST"

ssh $DEST -C -L 5900:127.0.0.1:5900 -fN

echo "Done."
