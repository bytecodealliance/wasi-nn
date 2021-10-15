#!/bin/bash

# Sets up the AssemblyScript demo for wasi-nn. Note you'll need to build and copy the wasmtime CLI
# and copy it to this directory for this script to work.

set -e
WASMTIME_DIR=$(dirname "$0" | xargs dirname)
DOWNLOAD_DIR=$WASMTIME_DIR/build
FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet

# Download all necessary test fixtures to the temporary directory.
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/mobilenet.bin
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/mobilenet.xml
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/tensor-1x224x224x3-f32.bgr
cp -rn images $DOWNLOAD_DIR

if [ ! -f $DOWNLOAD_DIR/images/0.jpg ]; then
    wget http://images.cocodataset.org/test-stuff2017/000000003188.jpg -O $DOWNLOAD_DIR/images/0.jpg
fi
if [ ! -f $DOWNLOAD_DIR/images/1.jpg ]; then
    wget http://images.cocodataset.org/test-stuff2017/000000001371.jpg -O $DOWNLOAD_DIR/images/1.jpg
fi
if [ ! -f $DOWNLOAD_DIR/images/2.jpg ]; then
    wget http://images.cocodataset.org/test-stuff2017/000000002288.jpg -O $DOWNLOAD_DIR/images/2.jpg
fi
if [ ! -f $DOWNLOAD_DIR/images/3.jpg ]; then
    wget http://images.cocodataset.org/test-stuff2017/000000002365.jpg -O $DOWNLOAD_DIR/images/3.jpg
fi
if [ ! -f $DOWNLOAD_DIR/images/4.jpg ]; then
    wget http://images.cocodataset.org/test-stuff2017/000000001643.jpg -O $DOWNLOAD_DIR/images/4.jpg
fi

# Run the demo
wasmtime run build/optimized.wasm --dir build --wasi-modules=experimental-wasi-nn
