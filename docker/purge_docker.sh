#!/bin/bash

set -x
set -e

docker system prune -a

docker system df

exit 0


sudo service docker stop

rm -rf /var/lib/docker

docker daemon --storage-opt dm.basesize=20G

sudo service docker start

docker info


