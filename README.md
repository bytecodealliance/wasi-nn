<div align="center">
  <h1><code>wasi-nn</code></h1>

  <strong>A <a href="https://bytecodealliance.org/">Bytecode Alliance</a> project</strong>

  <p><strong>High-level bindings for writing wasi-nn applications</strong></p>

  <p>
    <a href="https://github.com/bytecodealliance/wasi-nn/actions?query=workflow%3ACI">
      <img src="https://github.com/bytecodealliance/wasi-nn/workflows/CI/badge.svg" alt="CI status"/>
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

[crates.io]: https://crates.io/crates/wasi-nn
[Rust README]: rust/README.md
[npmjs.com]: https://www.npmjs.com/package/wasi-nn
[AssemblyScript README]: assemblyscript/README.md


### Examples

This repository includes examples of using these bindings. See the [Rust example] and
[AssemblyScript example] to walk through an end-to-end image classification using an AlexNet model.
Run them with:
 - `./build.sh rust` runs the [Rust example]
 - `./build.sh as` runs the [AssemblyScript example]

[Rust example]: rust/examples/classification-example
[AssemblyScript example]: assemblyscript/examples/object-classification.ts


### Related Links

- [WASI]
- [wasi-nn]
- [Wasmtime]
- [AssemblyScript]

[WASI]: https://github.com/WebAssembly/WASI
[AssemblyScript]: https://www.assemblyscript.org/


### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
