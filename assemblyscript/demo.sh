#!/bin/bash

# Sets up the AssemblyScript demo for wasi-nn. Note you'll need to build and copy the wasmtime CLI 
# and copy it to this directory for this script to work.

set -e
DOWNLOAD_DIR="$(dirname "$0" | xargs dirname)/build"
WASMTIME_DIR=$(dirname "$0" | xargs dirname)
FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/alexnet

# Download all necessary test fixtures to the temporary directory.
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/alexnet.bin
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/alexnet.xml
wget --no-clobber --directory-prefix=$DOWNLOAD_DIR $FIXTURE/tensor-1x3x227x227-f32.bgr

# Run the demo
./wasmtime run build/optimized.wasm --dir build
