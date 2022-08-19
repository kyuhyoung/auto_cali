#!/bin/bash
########################################################################################################
: << 'END'
#   build
cd docker_file/;    sudo docker build --force-rm --shm-size=64g -t u_22_python_opencv -f Dockerfile_u_22_python_opencv .;   cd -
END
########################################################################################################
#: << 'END'
sudo docker run --rm -it --shm-size=64g -p 9011:9011 -e DISPLAY=$DISPLAY -v $PWD:/workspace/auto_cali -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix.rw u_22_python_opencv fish 
#END
########################################################################################################
