#!/bin/bash

set -e
if [ -z $1 ]; then
    echo "Please specify as or rust to build"
else
    BUILD_TYPE=$1
    WASI_NN_DIR=$(dirname "$0" | xargs dirname)
    WASI_NN_DIR=$(realpath $WASI_NN_DIR)
    source /opt/intel/openvino/bin/setupvars.sh

    case $BUILD_TYPE in
        as)
        pushd $WASI_NN_DIR/assemblyscript
            npm install
            npm run demo
        ;;

        rust)
            echo "The first argument: $1"
            FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
            pushd $WASI_NN_DIR/rust/
            cargo build --release --target=wasm32-wasi
            mkdir -p $WASI_NN_DIR/rust/examples/classification-example/build
            RUST_BUILD_DIR=$(realpath $WASI_NN_DIR/rust/examples/classification-example/build/)
            pushd examples/classification-example
            cargo build --release --target=wasm32-wasi
            cp target/wasm32-wasi/release/wasi-nn-example.wasm $RUST_BUILD_DIR
            pushd build
            wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.bin
            wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.xml
            wget --no-clobber --directory-prefix=$RUST_BUILD_DIR $FIXTURE/tensor-1x224x224x3-f32.bgr
            wasmtime run --mapdir fixture::$RUST_BUILD_DIR wasi-nn-example.wasm
        ;;
        *)
            echo "Unknown build type $BUILD_TYPE"
        ;;
    esac
fi


