#!/bin/bash

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
        echo -e "\033[1;32m[WARNING] $@\033[0m"
    else
        echo "$@"
    fi
}


