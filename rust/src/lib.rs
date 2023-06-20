//! # WASI-NN
//!
//! ## Introduction
//! This project provides high-level wasi-nn bindings for Rust.
//! The basic idea: write your machine learning application in a high-level language using these bindings,
//! compile it to WebAssembly, and run it in a WebAssembly runtime that supports the [wasi-nn proposal](https://github.com/WebAssembly/wasi-nn/).
//!
//! ## Quick Start
//! ```rust
//! use wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget, TensorType};
//!
//! fn test(model_path: &'static str) -> Result<(), wasi_nn::Error> {
//!     // prepare input and output buffer.
//!     let input = vec![0f32; 224 * 224 * 3];
//!     let input_dim = vec![1, 224, 224, 3];
//!     // the input and output buffer can be any sized type, such as u8, f32, etc.
//!     let mut output_buffer = vec![0f32; 1001];
//!
//!     // build a tflite graph from file.
//!     let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU).build_from_files([model_path])?;
//!     // init graph execution context for this graph.
//!     let mut ctx = graph.init_execution_context()?;
//!     // set input
//!     ctx.set_input(0, TensorType::F32, &input_dim, &input)?;
//!     // do inference
//!     ctx.compute()?;
//!     // copy output to buffer
//!     let output_bytes = ctx.get_output(0, &mut output_buffer)?;
//!
//!     assert_eq!(output_bytes, output_buffer.len() * std::mem::size_of::<f32>());
//!     Ok(())
//! }
//! ```
//!
//! ## Note
//! This crate is experimental and will change to adapt the upstream
//! [wasi-nn proposal](https://github.com/WebAssembly/wasi-nn/).
//!
//! Now version is based on git commit ```f47f35c00c946cb0e3229f11f288bda9d3d12cff```
//!

#[allow(unused)]
mod generated;

mod error;
mod graph;
mod tensor;

pub use error::Error;
pub use graph::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding, GraphExecutionContext};
pub use tensor::TensorType;
