# WASI-NN for AssemblyScript

This package contains API bindings for [wasi-nn] system calls in AssemblyScript. It is similar in purpose to the [wasi bindings] but this package provides access to the optional neural network functionality from WebAssembly.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[wasi bindings]: https://github.com/bytecodealliance/wasi

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes in the wasi-nn
> specification.

### Use
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

### Examples
---

The examples demonstrate how to use wasi-nn from an AssemblyScript program.

For info on how to use Wasmtime to run the examples, see https://docs.wasmtime.dev/wasm-assemblyscript.html.

Filesystem functionality is provided by the as-wasi package, see https://github.com/jedisct1/as-wasi.
