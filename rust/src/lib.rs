wit_bindgen::generate!("wasi:nn/ml" in "wit/wasi-nn.wit");

pub use wasi::nn::errors;
pub use wasi::nn::graph;
pub use wasi::nn::tensor;
pub use wasi::nn::inference;