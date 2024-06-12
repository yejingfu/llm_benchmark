#!/bin/bash

LOG_SUPPRESS_WARN=0

LOG() {
    if [ x"$1" = x"INFO" ]; then
        shift
        echo -e "\033[1;36m[INFO] $@\033[0m"
    elif [ x"$1" = x"ERR" ]; then
        shift
        echo -e "\033[1;31m[ERROR] $@\033[0m"
        exit
    elif [ x"$1" = x"WARN" ]; then
        shift
        if [ $LOG_SUPPRESS_WARN -eq 0 ]; then
            echo -e "\033[1;32m[WARNING] $@\033[0m"
        fi
    else
        echo "$@"
    fi
}

check_image_exists() {
    local image=$1
    if [ x"$image" = x"" ]; then
        LOG ERR "The docker image is empty"
    fi
    images=`docker images $image`
    name=(${image//:/ })
    if [[ $images =~ "${name[0]}" ]]; then
        echo 1
    else
        echo 0
    fi
}

check_container_exists() {
    local name=$1
    docker ps -a | grep $name > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo 1
    else
        echo 0
    fi
}

remove_docker_container() {
    local name=$1
    if [ x"$name" = x"" ]; then
        LOG ERR "Cannot remove container with an empty name"
    fi
    local ret=$(check_container_exists $name)
    while [[ $ret -eq 1 ]]
    do
        # docker stop $name
        docker rm -f $name
        ret=$(check_container_exists $name)
    done
}

launch_docker_container() {
    local image=$1
    local name=$2
    local args=$3
    local cmd_params=$4
    local image_exists=$(check_image_exists $image)
    if [ $image_exists -eq 0 ]; then
        LOG ERR "The image does not exist: $image"
    fi
    if [ x"$name" = x"" ]; then
        LOG ERR "The container name is empty"
    fi
    if [ x"$args" = x"" ]; then
        LOG ERR "The launch argument is empty"
    fi
    docker ps -a | grep $name
    if [ $? -eq 0 ]; then
        LOG ERR "The docker container is alreay running: $name"
    fi
    docker run $args $image $cmd_params
}

