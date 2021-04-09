<div align="center">
  <h1><code>wasi-nn</code></h1>

  <strong>A <a href="https://bytecodealliance.org/">Bytecode Alliance</a> project</strong>

  <p>
    <strong>WASI-NN for standalone WASM applications</strong>
  </p>

  <p>
    <a href="https://github.com/bytecodealliance/wasi-nn/actions?query=workflow%3ACI">
      <img src="https://github.com/bytecodealliance/wasi-nn/workflows/CI/badge.svg" alt="CI status"/>
    </a>
    <a href="https://www.npmjs.com/package/wasi-nn">
      <img src="https://img.shields.io/npm/v/wasi-nn.svg"/>
    </a>
  </p>

</div>

## Introduction
---
The purpose of this project is to provide WASI-NN bindings for Rust and AssemblyScript. The compiled WASM code can then be run by a runtime that supports wasi-nn, such as [wasmtime](https://wasmtime.dev/).

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes in the wasi-nn
> specification.

## Use
---
[Rust README](rust/README.md)

[AssemblyScript README](assemblyscript/README.md)

## Examples
---
### Image classification:

[Rust](rust/examples/classification-example)

[AssemblyScript](assemblyscript/examples/object-classification.ts)

To build and run the image classification example use `./build.sh rust` for the Rust version or `./build.sh as` for AssemblyScript

## Related Links
---
[WASI](https://github.com/WebAssembly/WASI)

[Neural Network proposal for WASI](https://github.com/WebAssembly/wasi-nn)

[Wasmtime](https://wasmtime.dev/)

[AssemblyScript](https://www.assemblyscript.org/)