<div align="center">
  <h1><code>wasi-nn</code></h1>

  <strong>A <a href="https://bytecodealliance.org/">Bytecode Alliance</a> project</strong>

  <p><strong>High-level bindings for writing wasi-nn applications</strong></p>

  <p>
    <a href="https://github.com/bytecodealliance/wasi-nn/actions?query=workflow%3ACI">
      <img src="https://github.com/bytecodealliance/wasi-nn/workflows/CI/badge.svg" alt="CI status"/>
    </a>
    <a href="https://crates.io/crates/wasi-nn">
      <img src="https://img.shields.io/crates/v/wasi-nn.svg"/>
    </a>
    <a href="https://www.npmjs.com/package/as-wasi-nn">
      <img src="https://img.shields.io/npm/v/as-wasi-nn.svg"/>
    </a>
  </p>

</div>

### Introduction

This project provides high-level wasi-nn bindings for Rust and AssemblyScript. The basic idea: write
your machine learning application in a high-level language using these bindings, compile it to
WebAssembly, and run it in a WebAssembly runtime that supports the [wasi-nn] proposal, such as
[Wasmtime].

[Wasmtime]: https://wasmtime.dev
[wasi-nn]: https://github.com/WebAssembly/wasi-nn

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes
> in the [wasi-nn] specification.


### Use

 - In Rust, download the [crate from crates.io][crates.io] by adding `wasi-nn = "0.1"` as a Cargo
   dependency; more information in the [Rust README].
 - In AssemblyScript, download the [package from npmjs.com][npmjs.com] by adding `"as-wasi-nn":
   "^0.1"` as an NPM dependency; more information in the [AssemblyScript README].
 - When you call wasmtime, you'll need to pass the flag `--wasi-modules=experimental-wasi-nn` to enable the use use of wasi-nn.

[crates.io]: https://crates.io/crates/wasi-nn
[Rust README]: rust/README.md
[npmjs.com]: https://www.npmjs.com/package/wasi-nn
[AssemblyScript README]: assemblyscript/README.md

### Examples

This repository includes examples of using these bindings. See the [Rust example] and
[AssemblyScript example] to walk through an end-to-end image classification using an MobileNet model. The examples use OpenVino or TensorFlow as the backend. If you are running Ubuntu, you can simply run the script to install the supported version`.github/actions/install-openvino/install.sh`. Otherwise you'll need to visit the [Installation Guides] and follow the instructions for your OS. The version of OpenVino currently supported is openvino_2020.4.287. For TensorFlow, visit [Install TensorFlow for C](https://www.tensorflow.org/install/lang_c)

Once you have OpenVino installed, run them with:
 - `./build.sh rust openvino` runs the [Rust example] with OpenVino
 - `./build.sh rust tensorflow` runs the [Rust example] with TensorFlow
 - `./build.sh as openvino` runs the [AssemblyScript example] with OpenVino
 - `./build.sh as tensorflow` runs the [AssemblyScript example] with Tensorflow

If you want to see performance numbers, you can add `perf` to the end. For example, `./build.sh rust openvino perf`.  __NOTE__: this currently only works for the Rust example.

[Rust example]: rust/examples/classification-example
[AssemblyScript example]: assemblyscript/examples/object-classification.ts
[Installation Guides]: https://docs.openvinotoolkit.org/latest/installation_guides.html

### Related Links

- [WASI]
- [wasi-nn]
- [Wasmtime]
- [AssemblyScript]
- [OpenVino]

[WASI]: https://github.com/WebAssembly/WASI
[AssemblyScript]: https://www.assemblyscript.org/
[OpenVino]: https://docs.openvinotoolkit.org/latest/index.html

### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
