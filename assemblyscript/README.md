# wasi-nn bindings for AssemblyScript

This package contains high-level AssemblyScript bindings for [wasi-nn] system calls. It is similar
in purpose to the [as-wasi] bindings but this package provides optional access to a system's machine
learning functionality from WebAssembly.

[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[as-wasi]: https://github.com/bytecodealliance/wasi

> __NOTE__: These bindings are experimental (use at your own risk) and subject to upstream changes
> in the [wasi-nn] specification.

### Use

1. Add the dependency for wasi-nn to your `package.json`:
  ```
  "dependencies": {
    "as-wasi-nn": "0.2.1"
  }
  ```

2. Import the objects and functions you want to use in your project:
  ```
  import { Graph, Tensor, TensorType, GraphEncoding, ExecutionTarget } from "wasi-nn";
  ```

3. Compile your application to WebAssembly (see AssemblyScript's [quick start]).

4. Run the generated WebAssembly in a runtime supporting [wasi-nn], e.g. [Wasmtime].

[Wasmtime]: https://wasmtime.dev


### Build

To build this package from source, run `npm run asbuild`. Compiling your AssemblyScript application
to WebAssembly is a topic best covered by AssemblyScript's [quick start].

[quick start]: https://www.assemblyscript.org/quick-start.html


### Examples

The included [examples] demonstrate how to use wasi-nn from an AssemblyScript program. Run them
with:

```
npm run demo
```

For more information info on how to use Wasmtime to run the examples, see Wasmtime's [AssemblyScript
documentation]. Note that the wasi-nn bindings do not perform any WASI filesystem access; this
functionality is provided by the [as-wasi] package.

[examples]: examples
[AssemblyScript documentation]: https://docs.wasmtime.dev/wasm-assemblyscript.html
[as-wasi]: https://github.com/jedisct1/as-wasi


### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: ../LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
