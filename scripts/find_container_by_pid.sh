#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/base.sh

function usage() {
    LOG INFO "$PRG_NAME <pid>"
    exit
}

function main() {
    if [ "$#" -eq 0 ];then
        usage
    fi
    local pid=$1
    LOG INFO "Find container by process ID: ${pid}"
    containers=$(docker ps --format {{.Names}})
    local found=
    for con in ${containers[@]}; do
        # ret=$(docker top ${con} | grep  $pid)
        docker top ${con} | grep  $pid > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            found=${con}
            break
        fi
    done
    if [ x"${found}" != x"" ]; then
        LOG INFO "Find container: ${found}"
    else
        LOG INFO "No container found"
    fi
}

main "$@"

