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
  </p>

</div>

## Introduction
---
The purpose of this project is to provide WASI-NN bindings for Rust and AssemblyScript. The compiled WASM code can then be run by a runtime that supports wasi-nn, such as [wasmtime](https://wasmtime.dev/).

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes in the wasi-nn
> specification.

## Use
---
> __NOTE__: The wasi-nn binding packages have not been published yet. Local versions will need to be used until then.
>
### For Rust
---
Add the dependency for wasi-nn to your `Cargo.toml`:

```toml
[dependencies]
wasi-nn = "0.1.0"
```

Use the wasi-nn APIs in your application:

```rust
use wasi_nn;

unsafe {
    wasi_nn::load(
        &[&xml.into_bytes(), &weights],
        wasi_nn::GRAPH_ENCODING_OPENVINO,
        wasi_nn::EXECUTION_TARGET_CPU,
    )
    .unwrap()
}
```

Compile the application to WebAssembly:

```shell script
cargo build --target=wasm32-wasi
```

### For AssemblyScript
---
Add the dependency for wasi-nn to your `package.json`:
```
"dependencies": {
  "wasi-nn": "0.1.0"
}
```

Import the objects and functions you want to use in your project:
```
import { Graph, Tensor, TensorType, GraphEncoding, ExecutionTarget } from "wasi-nn";
```
## Examples
---
### Image classification:

[Rust](rust/examples/classification-example)

[AssemblyScript](assembly/examples/object-classification.ts)
## Related Links
---
[WASI](https://github.com/WebAssembly/WASI)

[Neural Network proposal for WASI](https://github.com/WebAssembly/wasi-nn)

[Wasmtime](https://wasmtime.dev/)

[AssemblyScript](https://www.assemblyscript.org/)