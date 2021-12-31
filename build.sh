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
        PERF=$3

        export BACKEND=$BACKEND
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
                        FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
                        wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.bin
                        wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.xml
                        ;;
                    tensorflow)
                        echo "Using Tensorflow"
                        cp src/saved_model.pb $RUST_BUILD_DIR
                        cp -r src/variables $RUST_BUILD_DIR
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

