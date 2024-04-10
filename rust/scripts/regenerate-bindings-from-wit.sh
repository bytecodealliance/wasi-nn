#!/usr/bin/env bash
set -e

# This script regenerates the Rust bindings for the wasi-nn specification. It has several steps:
#  1. retrieve and build `wit-bindgen`
#  2. retrieve the wasi-nn specification as WIT
#  3. use the WIT file to overwrite the `src/generated.rs` file
#
# Usage: $ regenerate-bindings-from-wit.sh
#
# The following environment variables can be overriden from the command line. Note that `*_REVISION`
# variables accept a commit hash, a branch, a tag, etc.
WIT_BINDGEN_REPOSITORY=${WIT_BINDGEN_REPOSITORY:-https://github.com/bytecodealliance/wit-bindgen.git}
WIT_BINDGEN_REVISION=${WIT_BINDGEN_REVISION:-main}
WASI_NN_RAW_URL=${WASI_NN_RAW_URL:-https://raw.githubusercontent.com/WebAssembly/wasi-nn}
WASI_NN_REVISION=${WASI_NN_REVISION:-main}

echo "=== Retrieve and build 'wit-bindgen' ==="
TMP_DIR=$(mktemp -d /tmp/regenerate-bindings.XXXXXX)
pushd $TMP_DIR
# This block attempts to retrieve the least amount of Git history while allowing the user to pick
# any revision.
git init
git remote add origin ${WIT_BINDGEN_REPOSITORY}
git fetch --depth 1 origin ${WIT_BINDGEN_REVISION}
git checkout FETCH_HEAD
git submodule update --init --depth 1
cargo build --bin wit-bindgen --release 
popd

echo
echo "=== Retrieve the wasi-nn specification as WIT ==="

curl ${WASI_NN_RAW_URL}/${WASI_NN_REVISION}/wit/wasi-nn.wit --output ${TMP_DIR}/wasi-nn.wit

echo "=== Overwrite the 'src/generated.rs' file ==="
# From https://stackoverflow.com/a/246128.
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
GENERATED_RS=$(realpath ${SCRIPT_DIR}/../src/generated.rs)
${TMP_DIR}/target/release/wit-bindgen rust ${TMP_DIR}/wasi-nn.wit --out-dir ${TMP_DIR}
cp ${TMP_DIR}/ml.rs ${GENERATED_RS}

# Clean up
rm -rf $TMP_DIR
