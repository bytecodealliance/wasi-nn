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
 *   (import "wasi_ephemeral_nn" "load" (func (param U32 U32 U32 U32 U32) (result U32)))
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

export declare function load(builder_ptr: u32, builder_len: u32, encoding: u32, target: u32, out_graph: u32): u32;
export declare function init_execution_context(graph: u32, context: u32): u32;
export declare function set_input(context: u32, index: u32, tensor: u32): u32;
export declare function compute(context: u32): u32;
export declare function get_output(context: u32, index: u32, out_buffer: u32, buffer_max_size: u32, out_bytes_written: i32): u32;
export declare function convert_image(path_ptr: u32, path_len: u32, width: u32, height: u32, precision: u32, out_buffer: u32, out_buffer_len: u32, out_bytes_written: i32): u32;