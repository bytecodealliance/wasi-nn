#!/bin/bash

set -e
if [ ! -d "/opt/intel/openvino" ]; then
    echo "Please install OpenVino"

else
    while getopts b:m:l:t:o:c:s:e:z:j: flag
    do
        case "${flag}" in
            b) BACKEND=${OPTARG};;
            m) MODEL=${OPTARG};;
            l) LOOP_SIZE=${OPTARG};;
            t) BUILD_TYPE=${OPTARG};;
            o) OUT_DIR=${OPTARG};;
            c) CPU_INFO=${OPTARG};;
            s) CPU_START=${OPTARG};;
            e) CPU_END=${OPTARG};;
            z) BATCH_SZ=${OPTARG};;
            j) THREAD_JMP=${OPTARG};;
        esac
    done

    # Default values
    if [ -z "$BACKEND" ]; then BACKEND="openvino"; fi
    if [ -z "$MODEL" ]; then MODEL="mobilenet_v2"; fi
    if [ -z "$LOOP_SIZE" ]; then LOOP_SIZE="1"; fi
    if [ -z "$BUILD_TYPE" ]; then BUILD_TYPE="rust"; fi
    if [ -z "$OUT_DIR" ]; then OUT_DIR="RESULTS"; fi
    if [ -z "$CPU_INFO" ]; then CPU_INFO="UNKOWN"; fi
    if [ -z "$BATCH_SZ" ]; then BATCH_SZ=1; fi
    if [ -z "$THREAD_JMP" ]; then THREAD_JMP=8; fi
    if [ -z "$CPU_START" ] && [ ! -z "$CPU_END" ]
    then
        echo "Error: CPU_END defined but CPU_START was not"
        return;
    else
        if [ ! -z "$CPU_START" ]
        then
            THREADS=$(($CPU_END - $CPU_START))

            # Account for starting with CPU 0
            if [ $CPU_START -eq 0 ]; then THREADS=$(($THREADS + 1)); fi

            # OpenVINO will not use more than 1/2 the available threads, so if you specify more it doesn't matter.
            if [ $THREADS -gt $(("$(grep -c processor /proc/cpuinfo)" / 2 )) ]
            then
                THREADS=$(("$(grep -c processor /proc/cpuinfo)" / 2 ))
            fi
        else
            THREADS=$(("$(grep -c processor /proc/cpuinfo)" / 2 ))
        fi
    fi

    export BACKEND=$BACKEND
    export MODEL=$MODEL
    export LOOP_SIZE=$LOOP_SIZE
    export BUILD_TYPE=$BUILD_TYPE
    export OUT_DIR=$OUT_DIR
    export CPU_INFO=$CPU_INFO
    export THREADS=$THREADS
    export CPU_END=$CPU_END

    WASI_NN_DIR=$(dirname "$0" | xargs dirname)
    WASI_NN_DIR=$(realpath $WASI_NN_DIR)

    case $BUILD_TYPE in
        as)

            pushd $WASI_NN_DIR/assemblyscript
            npm install

            case $BACKEND in
                openvino)
                    npm run openvino
                    ;;
                tensorflow)
                    npm run tensorflow
                    ;;
                *)
                    echo "Unknown backend, please enter 'openvino' or 'tensorflow'"
                    exit;
                    ;;
            esac
            ;;

        rust)
            pushd $WASI_NN_DIR/rust/
            cargo build --release --target=wasm32-wasi
            mkdir -p $WASI_NN_DIR/rust/examples/classification-example/build
            RUST_BUILD_DIR=$(realpath $WASI_NN_DIR/rust/examples/classification-example/build/)
            cp -rn images $RUST_BUILD_DIR
            pushd examples/classification-example
            export MAPDIR="fixture"
            cargo build --release --target=wasm32-wasi
            cp target/wasm32-wasi/release/wasi-nn-example.wasm $RUST_BUILD_DIR

            case $BACKEND in
                openvino)
                    echo "Using OpenVino"
                    source /opt/intel/openvino/bin/setupvars.sh
                    RUST_DIR=$(realpath $WASI_NN_DIR/rust/examples/classification-example/)
                    FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures

                    if [ ! -f "models/mobilenet_v2/model.bin" ]
                    then
                        wget -O models/mobilenet_v2/model.bin --no-clobber $FIXTURE/mobilenet/mobilenet.bin
                        wget -O models/mobilenet_v2/model.xml --no-clobber $FIXTURE/mobilenet/mobilenet.xml
                    fi
                    if [ ! -f "models/inception_v3/model.bin" ]
                    then
                        wget -O models/inception_v3/model.bin --no-clobber $FIXTURE/inception/inception.bin
                        wget -O models/inception_v3/model.xml --no-clobber $FIXTURE/inception/inception.xml
                    fi

                    cp models/$MODEL/model.bin $RUST_BUILD_DIR
                    cp models/$MODEL/model.xml $RUST_BUILD_DIR
                    cp models/$MODEL/tensor.desc $RUST_BUILD_DIR
                    ;;
                tensorflow)
                    echo "Using Tensorflow"
                    cp src/saved_model.pb $RUST_BUILD_DIR
                    cp -r src/variables $RUST_BUILD_DIR
                    cp models/$MODEL/tensor.desc $RUST_BUILD_DIR
                    ;;
                *)
                    echo "Unknown backend, please enter 'openvino' or 'tensorflow'"
                    exit;
                    ;;
            esac

            pushd build
            # Save results to the out_dir
            echo "Making dir $OUT_DIR"
            mkdir -p "$OUT_DIR/SUMMARY"

            if [ ! -z "$CPU_START" ] && [ ! -z "$CPU_END" ]
            then
                current_end=$(($CPU_END))

                for i in $(seq 1 $BATCH_SZ)
                do
                    echo "$CPU_INFO,$BACKEND,$MODEL,$THREADS" >> testout_all.csv
                    echo "$CPU_INFO,$BACKEND,$MODEL,$THREADS" >> testout.csv
                    echo "USING CORES $CPU_START $current_end - Batch # $i of $BATCH_SZ"
                    taskset --cpu-list $CPU_START-$current_end wasmtime run --dir . --mapdir fixture::$RUST_BUILD_DIR  wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
                    cp testout_all.csv "$OUT_DIR/testout_all-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S%3N").csv"
                    cp testout.csv "$OUT_DIR/SUMMARY/testout-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S%3N").csv"
                    ls -l "$OUT_DIR"
                    rm testout.csv
                    rm testout_all.csv
                    current_end=$(($current_end + $THREAD_JMP))
                    CPU_END=$(($current_end + $THREAD_JMP))
                    THREADS=$(($current_end-$CPU_START+1))

                done
            else
                echo "$CPU_INFO,$BACKEND,$MODEL,$THREADS" >> testout_all.csv
                echo "$CPU_INFO,$BACKEND,$MODEL,$THREADS" >> testout.csv
                wasmtime run --dir . --mapdir fixture::$RUST_BUILD_DIR  wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
                # Save results to the out_dir
                cp testout_all.csv "$OUT_DIR/testout_all-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S%3N").csv"
                cp testout.csv "$OUT_DIR/SUMMARY/testout-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S%3N").csv"
                rm testout.csv
                rm testout_all.csv
            fi
        ;;
        *)
            echo "Unknown build type $BUILD_TYPE"
        ;;
    esac
fi

