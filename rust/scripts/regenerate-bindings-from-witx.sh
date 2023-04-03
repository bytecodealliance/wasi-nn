#!/usr/bin/env bash
set -e

# This script regenerates the Rust bindings for the wasi-nn specification. It has several steps:
#  1. retrieve and build `witx-bindgen`
#  2. retrieve the wasi-nn specification as WITX
#  3. use the WITX file to overwrite the `src/generated.rs` file
#
# Usage: $ regenerate-bindings-from-witx.sh
#
# The following environment variables can be overriden from the command line. Note that `*_REVISION`
# variables accept a commit hash, a branch, a tag, etc.
WITX_BINDGEN_REPOSITORY=${WITX_BINDGEN_REPOSITORY:-https://github.com/bytecodealliance/wasi}
WITX_BINDGEN_REVISION=${WITX_BINDGEN_REVISION:-main}
WASI_NN_RAW_URL=${WASI_NN_RAW_URL:-https://raw.githubusercontent.com/WebAssembly/wasi-nn}
WASI_NN_REVISION=${WASI_NN_REVISION:-main}

echo "=== Retrieve and build 'witx-bindgen' ==="
TMP_DIR=$(mktemp -d /tmp/regenerate-bindings.XXXXXX)
pushd $TMP_DIR
# This block attempts to retrieve the least amount of Git history while allowing the user to pick
# any revision.
git init
git remote add origin ${WITX_BINDGEN_REPOSITORY}
git fetch --depth 1 origin ${WITX_BINDGEN_REVISION}
git checkout FETCH_HEAD
git submodule update --init --depth 1
cargo build --release -p witx-bindgen
popd

echo
echo "=== Retrieve the wasi-nn specification as WITX ==="
curl ${WASI_NN_RAW_URL}/${WASI_NN_REVISION}/wasi-nn.witx --output ${TMP_DIR}/wasi-nn.witx

echo
echo "=== Overwrite the 'src/generated.rs' file ==="
# From https://stackoverflow.com/a/246128.
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
GENERATED_RS=$(realpath ${SCRIPT_DIR}/../src/generated.rs)
${TMP_DIR}/target/release/witx-bindgen ${TMP_DIR}/wasi-nn.witx > ${GENERATED_RS}
# Also, here we fix up an issue in which `witx-bindgen` does not generate correct lifetimes; see
# https://github.com/bytecodealliance/wasi/issues/65.
sed -i "s/pub struct Tensor {/pub struct Tensor<'a> {/" ${GENERATED_RS}
sed -i "s/pub dimensions: TensorDimensions<'_>,/pub dimensions: TensorDimensions<'a>,/" ${GENERATED_RS}
sed -i "s/pub data: TensorData<'_>,/pub data: TensorData<'a>,/" ${GENERATED_RS}
sed -i "s/GraphBuilder<'_>/GraphBuilder<'a>/" ${GENERATED_RS}
sed -i "s/GraphBuilder<'_>/GraphBuilder<'a>/" ${GENERATED_RS}

# Clean up
rm -rf $TMP_DIR
