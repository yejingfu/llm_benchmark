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

install_docker() {
    ret=`docker -v`
    if  [[ $ret == *"Docker version"*  ]]; then
        return 1
    else
        LOG INFO "Install CUDA container toolkit"
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
        dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.0-550.54.14-1_amd64.deb
        cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
        apt-get update
        apt-get -y install cuda-toolkit-12-4
        apt-get install -y nvidia-container-toolkit
        return 1
    fi
}

check_image_exists() {
    local image=$1
    LOG INFO "Checking docker image: $image"
    if [ x"$image" = x"" ]; then
        LOG ERR "The docker image is empty"
        return 0
    fi
    images=`docker images $image`
    name=(${image//:/ })
    if [[ $images =~ "${name[0]}" ]]; then
        return 1
    else
        return 0
    fi
}

check_container_exists() {
    local name=$1
    if docker ps -a 2>/dev/null | grep -q "$name";then
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
    docker run $args --name $name $image $cmd_params
}

convert_to_array() {
    local str="$1"
    local IFS=','
    local -a arr=($str)
    echo ${arr[@]}
}

count_numbers() {
    local numbers_str="$1"
    local IFS=','
    local -a numbers=($numbers_str)
    echo ${#numbers[@]}
}

contains_value() {
    local value="$1"
    local -a arr=("${@:2}")
    for i in "${arr[@]}";do
        if [[ "$i" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

