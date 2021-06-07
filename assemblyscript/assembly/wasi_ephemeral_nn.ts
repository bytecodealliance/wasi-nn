/**
 * Declare the wasi-nn API for use in AssemblyScript. We do not expect users to access these
 * declarations directly--use the `wasi_nn.ts` module instead. If you must use these, write `import
 * { load } from "./wasi_ephemeral_nn";`. When compiled to WebAssembly, these functions will become
 * Wasm `import` declarations. They __must__ match the wasi-nn API exactly.
 *
 * To exactly match the wasi-nn API (codified in WITX at
 * https://github.com/WebAssembly/wasi-nn/blob/master/phases/ephemeral/witx/wasi_ephemeral_nn.witx),
 * we used some WIP tooling to display WAT signatures of WASI modules (see
 * https://github.com/WebAssembly/WASI/pull/377). Looking just at the first function, we know that
 * we need to expose a raw interface that looks something like:
 *
 *   (import "wasi_ephemeral_nn" "load" (func (param I32 I32 I32 I32 I32) (result I32)))
 *
 * It is interesting to note that the equivalent raw Rust interface (see
 * https://github.com/bytecodealliance/wasmtime/blob/main/crates/wasi-nn/examples/wasi-nn-rust-bindings/src/generated.rs#L163-L169),
 * looks like:
 *
 *   pub fn load(builder_ptr: *const GraphBuilder, builder_len: usize, encoding: GraphEncoding,
 *     target: ExecutionTarget, graph: *mut Graph) -> NnErrno;
 *
 * Note how `graph` is a mutable pointer, an "out" variable modified by the call. We denote these
 * here with the `out_` prefix.
 */

export declare function load(builder_ptr: i32, builder_len: i32, encoding: i32, target: i32, out_graph: i32): i32;
export declare function init_execution_context(graph: i32, context: i32): i32;
export declare function set_input(context: i32, index: i32, tensor: i32): i32;
export declare function compute(context: i32): i32;
export declare function get_output(context: i32, index: i32, out_buffer: i32, buffer_max_size: i32, out_bytes_written: i32): i32;
export declare function image_to_tensor(path_ptr: i32, path_len: i32, width: i32, height: i32, precision: i32, out_buffer: i32, buffer_max_size: i32): i32;
