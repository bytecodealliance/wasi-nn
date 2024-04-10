//! # `wasi-nn`
//!
//! ### Introduction
//!
//! This project provides high-level wasi-nn bindings for Rust. The basic idea: write your machine
//! learning application in a high-level language using these bindings, compile it to WebAssembly,
//! and run it in a WebAssembly runtime that supports the [wasi-nn
//! proposal](https://github.com/WebAssembly/wasi-nn/).
//!
//! ### Usage
//!
//! ```rust
//! use wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget, TensorType};
//!
//! fn test(model_path: &'static str) -> Result<(), wasi_nn::Error> {
//!     // Prepare input and output buffer; the input and output buffer can be any sized type, such
//!     // as u8, f32, etc.
//!     let input = vec![0f32; 224 * 224 * 3];
//!     let input_dim = vec![1, 224, 224, 3];
//!
//!     // Build a tflite graph from a file and set an input tensor.
//!     let graph = wasi_nn::graph::load(
//!            &builders,
//!            wasi_nn::graph::GraphEncoding::TensorflowLite,
//!            wasi_nn::graph::ExecutionTarget::Cpu,
//!     )?;
//!     let ctx = wasi_nn::inference::init_execution_context(graph)?;
//!     ctx.set_input(0, TensorType::F32, &input_dim, &input)?;
//!     let tensor = wasi_nn::tensor::Tensor {
//!         dimensions: input_dim,
//!         tensor_type: wasi_nn::tensor::TensorType::Fp32,
//!         data: input,
//!     };
//!
//!     wasi_nn::inference::set_input(context, 0, &tensor)?;
//!
//!     // Do the inference.
//!     wasi_nn::inference::compute(context)?;
//!
//!     // Copy output to abuffer.
//!     let output_bytes = wasi_nn::inference::get_output(context, 0)?;
//!     Ok(())
//! }
//! ```
//!
//! ### Note
//!
//! This crate is experimental and will change to adapt the upstream [wasi-nn
//! proposal](https://github.com/WebAssembly/wasi-nn/).
//!
//! Now version is based on git commit ```e2310b860db2ff1719c9d69816099b87e85fabdb```
//!

#[allow(unused)]
mod generated;

pub use crate::generated::wasi::nn::*;
