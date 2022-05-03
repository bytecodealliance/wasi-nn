#!/bin/bash

set -e
if [ ! -d "/opt/intel/openvino" ]; then
    echo "Please install OpenVino"

else
    if [ -z $1 ]; then
        echo "Please specify as or rust to build"
    else
        BUILD_TYPE=$1
        BACKEND=$2
        PERF=$4
        LOOP_SIZE=$5
        export BACKEND=$BACKEND

        if [ -z "$3" ]; then MODEL="mobilenet_v2"; else MODEL=$3; fi
        if [ -z "$5" ]; then LOOP_SIZE=1; else LOOP_SIZE=$5; fi
        export MODEL=$MODEL
        export LOOP_SIZE=$LOOP_SIZE
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
                echo "The first argument: $1"
                pushd $WASI_NN_DIR/rust/
                cargo build --release --target=wasm32-wasi
                mkdir -p $WASI_NN_DIR/rust/examples/classification-example/build
                RUST_BUILD_DIR=$(realpath $WASI_NN_DIR/rust/examples/classification-example/build/)
                cp -rn images $RUST_BUILD_DIR
                pushd examples/classification-example
                export MAPDIR="fixture"

                case $PERF in
                    perf)
                        case $LOOP_SIZE in
                            ''|*[!0-9]*)
                                echo "Loop size needs to be a number";
                                exit;
                                ;;
                        esac
                    echo "RUNNING PERFORMANCE CHECKS"
                        cargo build --release --target=wasm32-wasi --features performance
                        ;;
                    *)
                        cargo build --release --target=wasm32-wasi
                        ;;
                esac

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
                        if [ ! -f "models/alexnet/model.bin" ]
                        then
                            wget -O models/alexnet/model.bin --no-clobber $FIXTURE/alexnet/alexnet.bin
                            wget -O models/alexnet/model.xml --no-clobber $FIXTURE/alexnet/alexnet.xml
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
                wasmtime run --mapdir fixture::$RUST_BUILD_DIR  wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
            ;;
            *)
                echo "Unknown build type $BUILD_TYPE"
            ;;
        esac

    fi
fi

