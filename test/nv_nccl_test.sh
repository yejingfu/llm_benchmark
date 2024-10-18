#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)

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

NCCL_DIR=
NCCL_TESTS_DIR=
CUDA_SAMPLES_DIR=
LOG_FILE=
NUM_GPUS=(2 4 8)

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --nccl-dir The path to nccl library"
    LOG INFO "  --nccl-tests-dir The path to nccl-test tools"
    LOG INFO "    If nccl-tests not installed, please git clone and make from https://github.com/NVIDIA/nccl and https://github.com/NVIDIA/nccl-tests, example"
    LOG INFO "      git clone https://github.com/NVIDIA/nccl && cd nccl && make -j8"
    LOG INFO "      git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make NCCL_HOME=../nccl/build -j8"
    LOG INFO "  --cuda-samples-dir The path to cuda samples, which is used to test P2P latency, if not available, build from https://github.com/NVIDIA/cuda-samples.git"
    LOG INFO "  --log-file The output log file to save the result"
    exit
}

function run_all_reduce() {
    local num=$#
    if [ $num -eq 0 ]; then
        LOG ERR "Empty GPU id list"
    fi
    local perf_bin=${NCCL_TESTS_DIR}/build/all_reduce_perf
    local gpu_list=$1
    shift
    for (( n=1; n<$num; n++ ));do
        gpu_list="$gpu_list,$1"
        shift
    done
    LOG INFO "[RUN]: CUDA_VISIBLE_DEVICES=$gpu_list $perf_bin -b 1024 -e 1G -f 2 -g $num"
    if [ x"$LOG_FILE" != x"" ]; then
        echo "====[all_reduce]: $gpu_list" >> $LOG_FILE
        CUDA_VISIBLE_DEVICES=$gpu_list $perf_bin -b 1024 -e 1G -f 2 -g $num | tee -a $LOG_FILE
    else
        CUDA_VISIBLE_DEVICES=$gpu_list $perf_bin -b 1024 -e 1G -f 2 -g $num
    fi
}

function run() {
    LOG INFO "=== ${NCCL_DIR}"
    if [ ! -e "${NCCL_DIR}/build" ]; then
        LOG ERR "Invalid nccl directory: ${NCCL_DIR}/build"
    fi
    local perf_bin=${NCCL_TESTS_DIR}/build/all_reduce_perf
    if [ ! -f "${perf_bin}" ]; then
        LOG ERR "Invalid perf bin file: ${perf_bin}"
    fi
    export LD_LIBRARY_PATH=${NCCL_DIR}/build/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    for n in ${NUM_GPUS[@]};do
        if [ $n -eq 2 ]; then
            LOG INFO "======== Testing 2 GPUS ========"
            for i in {0..7};do
                for j in {0..7};do
                    if [ $i -ne $j ]; then
                        LOG INFO "Pair: [$i , $j]"
                        run_all_reduce $i $j
                    fi
                done
            done
        elif [ $n -eq 4 ]; then
            LOG INFO "======== Testing 4 GPUS ========"
            run_all_reduce 0 1 2 3
            run_all_reduce 4 5 6 7
        elif [ $n -eq 8 ]; then
            LOG INFO "======== Testing 8 GPUS ========"
            run_all_reduce 0 1 2 3 4 5 6 7
        fi
    done

    if [ x"$CUDA_SAMPLES_DIR" != x"" ]; then
        LOG INFO "Test P2P latency"
        local bandwidth_test="$CUDA_SAMPLES_DIR/Samples/1_Utilities/bandwidthTest/bandwidthTest"
        local p2p_latency_test="$CUDA_SAMPLES_DIR/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"
        if [ ! -e $bandwidth_test ]; then
            LOG ERR "The test bin does not exist: $bandwidth_test"
        fi
        if [ ! -e $p2p_latency_test ]; then
            LOG ERR "The test bin does not exist: $p2p_latency_test"
        fi
        local tmp=$CUDA_VISIBLE_DEVICES
        unset CUDA_VISIBLE_DEVICES
        LOG INFO "[RUN] $bandwidth_test --device=i"
        if [ x"$LOG_FILE" != x"" ]; then
            echo "====[bandwidth_test]: 0,1,2,3,4,5,6,7" >> $LOG_FILE
            for i in {0..7};do $bandwidth_test --device=$i ;done | tee -a $LOG_FILE
        else
            for i in {0..7};do $bandwidth_test --device=$i ;done
        fi
        LOG INFO "[RUN] $p2p_latency_test"
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        if [ x"$LOG_FILE" != x"" ]; then
            echo "====[p2p_latency_test]" >> $LOG_FILE
            $p2p_latency_test | tee -a $LOG_FILE
        else
            $p2p_latency_test
        fi
        if [ x"$tmp" != x"" ]; then
            export CUDA_VISIBLE_DEVICES=$tmp
        fi
    fi
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --nccl-dir)
        shift
        NCCL_DIR="$1"
        shift
        ;;
    --nccl-tests-dir)
        shift
        NCCL_TESTS_DIR="$1"
        shift
        ;;
    --cuda-samples-dir)
        shift
        CUDA_SAMPLES_DIR="$1"
        shift
        ;;
    --log-file)
        shift
        LOG_FILE="$1"
        shift
        ;;
    *)
        usage
        break
    esac
    done
    run
}

main "$@"

