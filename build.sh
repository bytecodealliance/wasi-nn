#!/bin/bash

set -e
if [ ! -d "/opt/intel/openvino" ]; then
    echo "Please install OpenVino"

else
    if [ -z $1 ]; then
        echo "Please specify as, as_i2t, rust, or rust_i2t to build"
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

            as_i2t_host)
            pushd $WASI_NN_DIR/assemblyscript
                npm install
                npm run demo_i2t_host
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
                wget -c --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.bin
                wget -c --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.xml
                mkdir -p images
                if [ ! -f $RUST_BUILD_DIR/images/0.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000003188.jpg -O images/0.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/1.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000001371.jpg -O images/1.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/2.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000002288.jpg -O images/2.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/3.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000002365.jpg -O images/3.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/4.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000001643.jpg -O images/4.jpg
                fi
                wasmtime run --mapdir fixture::$RUST_BUILD_DIR wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
            ;;

            rust_i2t_host)
                echo "The first argument: $1"
                FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
                pushd $WASI_NN_DIR/rust/
                cargo build --release --target=wasm32-wasi --features i2t_host
                mkdir -p $WASI_NN_DIR/rust/examples/classification-example/build
                RUST_BUILD_DIR=$(realpath $WASI_NN_DIR/rust/examples/classification-example/build/)
                pushd examples/classification-example
                cargo build --release --target=wasm32-wasi --features i2t_host
                cp target/wasm32-wasi/release/wasi-nn-example.wasm $RUST_BUILD_DIR
                pushd build
                wget -c --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.bin
                wget -c --directory-prefix=$RUST_BUILD_DIR $FIXTURE/mobilenet.xml
                mkdir -p images
                if [ ! -f $RUST_BUILD_DIR/images/0.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000003188.jpg -O images/0.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/1.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000001371.jpg -O images/1.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/2.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000002288.jpg -O images/2.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/3.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000002365.jpg -O images/3.jpg
                fi
                if [ ! -f $RUST_BUILD_DIR/images/4.jpg ]; then
                    wget http://images.cocodataset.org/test-stuff2017/000000001643.jpg -O images/4.jpg
                fi
                wasmtime run --mapdir fixture::$RUST_BUILD_DIR wasi-nn-example.wasm --wasi-modules=experimental-wasi-nn
            ;;
            *)
                echo "Unknown build type $BUILD_TYPE"
            ;;
        esac

    fi
fi

