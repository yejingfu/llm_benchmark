#!/bin/bash
PRG_NAME=$(basename "${BASH_SOURCE[0]}")
CUR_DIR=$(cd `dirname $0`;pwd)
source $CUR_DIR/../scripts/base.sh
# example
# ./launch_eval.sh --opencompass-src /ppio/abao/code/add_long_test/opencompass --with-docker on/off --v_tokenizer /ppio/models --config config.json --eval-degree dummy
EV_OPENCOMPASS_SRC=
EV_EVAL_DEGREE=
EV_CONFIG=
EV_DOCKER=
EV_TOKENIZER=

function usage() {
    LOG INFO "$PRG_NAME [options]"
    LOG INFO "  --opencompass-src  The path of opencompass source code"
    LOG INFO "  --config The config file. Note: Can only be located in the opencompass source directory"
    LOG INFO "  --eval-degree The evaluation degree "
    exit
}


#检查并选择 opencompass 版本
select_opencompass_image() {
    # 定义镜像名称和标签
    local image_name="opencompass"
    local tag_preferred="0.2.4"
    local tag_alternative="0.3"


    # 首先检查首选版本 opencompass:0.2.4 是否存在
    if docker image inspect "${image_name}:${tag_preferred}" &> /dev/null; then
        echo "${image_name}:${tag_preferred}"
        return
    fi

    # 如果首选版本不存在，检查备选版本 opencompass:0.3 是否存在
    if docker image inspect "${image_name}:${tag_alternative}" &> /dev/null; then
        echo "${image_name}:${tag_alternative}"
        return
    fi

    # 如果两个版本都不存在，则build 镜像
    LOG INFO "[RUN]: docker build -t ${image_name}:${tag_preferred} $EV_OPENCOMPASS_SRC"
    docker build -t "${image_name}:${tag_preferred}" "${EV_OPENCOMPASS_SRC}"

    echo "${image_name}:${tag_preferred}"
    return
}

check_data(){
    # 定义目标目录
    target_dir="${EV_OPENCOMPASS_SRC}/data"
    mkdir -p ${target_dir}

    # 使用 find 命令来查找目标目录下的所有子目录，不包括目标目录本身
    subdir_count=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)

    # 检查子目录数目
    if [ "$subdir_count" -lt 18 ]; then
        # 如果子目录数目小于18，执行解压操作
        unzip_path="${EV_OPENCOMPASS_SRC}/OpenCompassData-core-20240423.zip"
        unzip -oq "$unzip_path" -d "${EV_OPENCOMPASS_SRC}"
        LOG INFO "Extracted $unzip_path to $target_dir because there were less than 18 subdirectories."
    else
        LOG INFO "There are already $subdir_count subdirectories in $target_dir, no extraction performed."
    fi
}


function run() {
    if [ x"$EV_CONFIG" = x"" ]; then
        LOG ERR "The config file is not set"
    fi
    if [ x"$EV_OPENCOMPASS_SRC" = x"" ]; then
        LOG ERR "The path of opencompass source code is not set"
    fi

    # 解压opencompass数据集
    check_data
    if [ x"$EV_DOCKER" = x"on" ]; then
        selected_image=$(select_opencompass_image)
        LOG INFO "using docker image: ${selected_image}"
    fi

    local args="--config ${EV_CONFIG}"
    
    LOG INFO "--docker ${EV_DOCKER}"
    LOG INFO "--docker ${EV_TOKENIZER}"

    if [ x"$EV_DOCKER" = x"on" ]; then
        LOG INFO "[RUN]: docker run --rm -it -v ${EV_OPENCOMPASS_SRC}:/opencompass --entrypoint python3 ${selected_image} /opencompass/entry.py ${args}"
        docker run --rm -it --gpus all  --net=host -e https_proxy=$https_proxy -e http_proxy=$http_proxy -v ${EV_OPENCOMPASS_SRC}:/opencompass -v ${EV_TOKENIZER}:/ppio_models --entrypoint python3 ${selected_image} /opencompass/entry.py ${args} 
    else
        LOG INFO "[RUN]: python3 ${EV_OPENCOMPASS_SRC}/entry.py ${args}"
        python3 ${EV_OPENCOMPASS_SRC}/entry.py ${args}
    fi
}

function main() {
    if [ "$#" -eq 0 ]; then
        usage
    fi
    while [ "$#" -gt 0 ]; do
    case "$1" in
    --opencompass-src)
        shift
        EV_OPENCOMPASS_SRC="$1"
        shift
        ;;
    --config)
        shift
        EV_CONFIG="$1"
        shift
        ;;
    --with-docker)
        shift
        EV_DOCKER="$1"
        shift
        ;;
    --v_tokenizer)
        shift
        EV_TOKENIZER="$1"
        shift
        ;;
    --eval-degree)
        shift
        EV_EVAL_DEGREE="$1"
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
