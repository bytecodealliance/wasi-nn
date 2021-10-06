#!/bin/bash
mkdir -p examples

case $1 in
    i2t_host)
        # Copy files for using the host version of convert_image
        cp assembly/convert_image/host/as-wasi-nn.ts assembly
        cp assembly/convert_image/host/wasi_ephemeral_nn.ts assembly
        cp assembly/convert_image/host/object-classification.ts examples/object-classification.ts
    ;;
    *)
        # Copy files for using the local version with pre converted images
        cp assembly/convert_image/local/as-wasi-nn.ts assembly
        cp assembly/convert_image/local/wasi_ephemeral_nn.ts assembly
        cp assembly/convert_image/local/object-classification.ts examples/object-classification.ts
    ;;
esac