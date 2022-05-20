#!/bin/bash

set -e
if [ ! -d "/opt/intel/openvino" ]; then
    echo "Please install OpenVino"

else
    while getopts b:m:l:t:o:c: flag
    do
        case "${flag}" in
            b) BACKEND=${OPTARG};;
            m) MODEL=${OPTARG};;
            l) LOOP_SIZE=${OPTARG};;
            t) BUILD_TYPE=${OPTARG};;
            o) OUT_DIR=${OPTARG};;
            c) CPU_INFO=${OPTARG};;
            t) THREADS=${OPTARG}
        esac
    done

    totalcpu="$(grep -c processor /proc/cpuinfo)"
    ((half=totalcpu/2))
    # Default values
    if [ -z "$BACKEND" ]; then BACKEND="openvino"; fi
    if [ -z "$MODEL" ]; then MODEL="mobilenet_v2"; fi
    if [ -z "$LOOP_SIZE" ]; then LOOP_SIZE="1"; fi
    if [ -z "$BUILD_TYPE" ]; then BUILD_TYPE="rust"; fi
    if [ -z "$OUT_DIR" ]; then OUT_DIR="RESULTS"; fi
    if [ -z "$CPU_INFO" ]; then CPU_INFO="UNKOWN"; fi
    if [ -z "$THREADS" ]; then THREADS=$half; fi

    echo "BJONES THREADS! $THREADS"

    export BACKEND=$BACKEND
    export MODEL=$MODEL
    export LOOP_SIZE=$LOOP_SIZE
    export BUILD_TYPE=$BUILD_TYPE
    export OUT_DIR=$OUT_DIR
    export CPU_INFO=$CPU_INFO
    export THREADS=$THREADS

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
            wasmtime run --dir . --mapdir fixture::$RUST_BUILD_DIR  wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
            # Save results to the out_dir
            echo "BJONES making dir $OUT_DIR"
            mkdir -p $OUT_DIR
            # cp -a derp.csv "$OUT_DIR/derp-$(date +"%Y-%m-%d-%H%M%S").csv"
            cp testout_all.csv "$OUT_DIR/testout_all-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S").csv"
            cp testout.csv "$OUT_DIR/testout-$BACKEND-$MODEL-$(date +"%Y-%m-%d-%H%M%S").csv"
            # cp *.csv $OUT_DIR
        ;;
        *)
            echo "Unknown build type $BUILD_TYPE"
        ;;
    esac
fi

