use crate::{tensor::Tensor, Error, TensorType};

/// Describes the encoding of the graph. This allows the API to be implemented by various backends
/// that encode (i.e., serialize) their graph IR with different formats.
/// Now the available backends are `Openvino`, `Onnx`, `Tensorflow`, `Pytorch`, `TensorflowLite`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub enum GraphEncoding {
    Openvino = 0,
    Onnx,
    Tensorflow,
    Pytorch,
    TensorflowLite,
}

/// Define where the graph should be executed.
/// Now the available devices are `CPU`, `GPU`, `TPU`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub enum ExecutionTarget {
    CPU = 0,
    GPU,
    TPU,
}

/// Graph factory, which can be used in order to configure the properties of a new graph.
/// Methods can be chained on it in order to configure it.
/// * Default Graph Encoding: `Openvino`.
/// * Default Execution Target: `CPU`.
///
/// ### Examples
///
/// #### build a graph with default config ( `CPU` + `Openvino` )
/// ```rust
/// use wasi_nn::{GraphBuilder, GraphEncoding};
///
/// let xml = "./mobilenet.xml";
/// let weight = "./mobilenet.bin";
/// let graph = GraphBuilder::default().build_from_files([xml, weight])?;
/// ```
///
/// #### build a graph with onnx backend and gpu device target
/// ```rust
/// use wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget};
///
/// let model_file = "./test.onnx";
/// let graph = GraphBuilder::new(GraphEncoding::Onnx, ExecutionTarget::GPU).build_from_files([model_file])?;
/// ```
///
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    encoding: GraphEncoding,
    target: ExecutionTarget,
}

impl Default for GraphBuilder {
    /// Create default GraphBuild
    /// * Default Graph Encoding: `Openvino`.
    /// * Default Execution Target: `CPU`.
    #[inline(always)]
    fn default() -> Self {
        Self::new(GraphEncoding::Openvino, ExecutionTarget::CPU)
    }
}

impl GraphBuilder {
    /// Create a new [```GraphBuilder```].
    #[inline(always)]
    pub fn new(encoding: GraphEncoding, target: ExecutionTarget) -> Self {
        Self { encoding, target }
    }

    /// Set graph encoding.
    #[inline(always)]
    pub fn encoding(mut self, encoding: GraphEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Set graph execution target.
    #[inline(always)]
    pub fn execution_target(mut self, execution_target: ExecutionTarget) -> Self {
        self.target = execution_target;
        self
    }

    /// Set graph execution target to `CPU`.
    #[inline(always)]
    pub fn cpu(mut self) -> Self {
        self.target = ExecutionTarget::CPU;
        self
    }

    /// Set graph execution target to `GPU`.
    #[inline(always)]
    pub fn gpu(mut self) -> Self {
        self.target = ExecutionTarget::GPU;
        self
    }

    /// Set graph execution target to `TPU`.
    #[inline(always)]
    pub fn tpu(mut self) -> Self {
        self.target = ExecutionTarget::TPU;
        self
    }

    #[inline(always)]
    pub fn build_from_bytes<B>(self, bytes_array: impl AsRef<[B]>) -> Result<Graph, Error>
    where
        B: AsRef<[u8]>,
    {
        let graph_builder_array: Vec<&[u8]> =
            bytes_array.as_ref().iter().map(|s| s.as_ref()).collect();
        let graph_handle =
            syscall::load(graph_builder_array.as_slice(), self.encoding, self.target)?;
        Ok(Graph {
            build_info: self,
            graph_handle,
        })
    }

    #[inline(always)]
    pub fn build_from_files<P>(self, files: impl AsRef<[P]>) -> Result<Graph, Error>
    where
        P: AsRef<std::path::Path>,
    {
        let mut graph_contents = Vec::with_capacity(files.as_ref().len());
        for file in files.as_ref() {
            graph_contents.push(std::fs::read(file).map_err(Into::<Error>::into)?);
        }
        let graph_builder_array: Vec<&[u8]> = graph_contents.iter().map(|s| s.as_ref()).collect();
        let graph_handle =
            syscall::load(graph_builder_array.as_slice(), self.encoding, self.target)?;
        Ok(Graph {
            build_info: self,
            graph_handle,
        })
    }
}

/// An execution graph for performing inference (i.e., a model), which can create instances of [`GraphExecutionContext`].
///
/// ### Example
/// ```rust
/// use wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget};
///
/// let model_file = "./test.tflite";
/// // create a graph using `GraphBuilder`
/// let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU).build_from_files([model_file])?;
/// // create an execution context using this graph
/// let mut graph_exec_ctx = graph.init_execution_context()?;
/// // set input tensors
/// // ......
///
/// // compute the inference on the given inputs
/// graph_exec_ctx.compute()?;
///
/// // get output tensors and do post-processing
/// // ......
/// ```
pub struct Graph {
    build_info: GraphBuilder,
    graph_handle: syscall::GraphHandle,
}

impl Graph {
    /// Get the graph encoding.
    #[inline(always)]
    pub fn encoding(&self) -> GraphEncoding {
        self.build_info.encoding
    }

    /// Get the graph execution target.
    #[inline(always)]
    pub fn execution_target(&self) -> ExecutionTarget {
        self.build_info.target
    }

    /// Get the graph id.
    #[inline(always)]
    pub fn graph_id(&self) -> syscall::GraphHandle {
        self.graph_handle
    }

    /// Use this graph to create a new instances of [`GraphExecutionContext`].
    #[inline(always)]
    pub fn init_execution_context(&self) -> Result<GraphExecutionContext, Error> {
        let ctx_handle = syscall::init_execution_context(self.graph_handle)?;
        Ok(GraphExecutionContext {
            graph: self,
            ctx_handle,
        })
    }
}

/// Bind a [`Graph`] to the input and output for an inference.
pub struct GraphExecutionContext<'a> {
    graph: &'a Graph,
    ctx_handle: syscall::GraphExecutionContextHandle,
}

impl<'a> GraphExecutionContext<'a> {
    /// Get the [`Graph`] instance for this [`GraphExecutionContext`] instance.
    #[inline(always)]
    pub fn graph(&self) -> &Graph {
        self.graph
    }

    /// Get the execution context id.
    #[inline(always)]
    pub fn context_id(&self) -> syscall::GraphExecutionContextHandle {
        self.ctx_handle
    }

    /// Set input uses the `data`, not only [u8], but also [f32], [i32], etc.
    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), Error> {
        let data_slice = data.as_ref();
        let buf = unsafe {
            core::slice::from_raw_parts(
                data_slice.as_ptr() as *const u8,
                data_slice.len() * std::mem::size_of::<T>(),
            )
        };
        let tensor_for_call = Tensor::new(dimensions, tensor_type, buf);
        syscall::set_input(self.ctx_handle, index, tensor_for_call)
    }

    /// Compute the inference on the given inputs.
    #[inline(always)]
    pub fn compute(&mut self) -> Result<(), Error> {
        syscall::compute(self.ctx_handle)
    }

    /// Copy output tensor to `out_buffer`, return the out **byte size**.
    #[inline(always)]
    pub fn get_output<T: Sized>(&self, index: usize, out_buffer: &mut [T]) -> Result<usize, Error> {
        let out_buf = unsafe {
            core::slice::from_raw_parts_mut(
                out_buffer.as_mut_ptr() as *mut u8,
                out_buffer.len() * std::mem::size_of::<T>(),
            )
        };
        syscall::get_output(self.ctx_handle, index, out_buf)
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
mod syscall {
    use crate::generated::wasi_ephemeral_nn;
    use crate::{error::BackendError, tensor::Tensor, Error, ExecutionTarget, GraphEncoding};

    pub(crate) type GraphHandle = i32;
    pub(crate) type GraphExecutionContextHandle = i32;

    #[inline(always)]
    pub(crate) fn load(
        graph_builder_array: &[&[u8]],
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<GraphHandle, Error> {
        let mut graph_handle = 0;
        let res = unsafe {
            wasi_ephemeral_nn::load(
                graph_builder_array.as_ptr() as i32,
                graph_builder_array.len() as i32,
                encoding as i32,
                target as i32,
                &mut graph_handle as *mut _ as i32,
            )
        };

        if res == 0 {
            Ok(graph_handle)
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[inline(always)]
    pub(crate) fn init_execution_context(
        graph_handle: GraphHandle,
    ) -> Result<GraphExecutionContextHandle, Error> {
        let mut ctx_handle = 0;
        let res = unsafe {
            wasi_ephemeral_nn::init_execution_context(
                graph_handle,
                &mut ctx_handle as *mut _ as i32,
            )
        };

        if res == 0 {
            Ok(ctx_handle)
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[inline(always)]
    pub(crate) fn set_input(
        ctx_handle: GraphExecutionContextHandle,
        index: usize,
        tensor: Tensor,
    ) -> Result<(), Error> {
        let res = unsafe {
            wasi_ephemeral_nn::set_input(ctx_handle, index as i32, &tensor as *const _ as i32)
        };
        if res == 0 {
            Ok(())
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[inline(always)]
    pub(crate) fn compute(ctx_handle: GraphExecutionContextHandle) -> Result<(), Error> {
        let res = unsafe { wasi_ephemeral_nn::compute(ctx_handle) };
        if res == 0 {
            Ok(())
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[inline(always)]
    pub(crate) fn get_output(
        ctx_handle: GraphExecutionContextHandle,
        index: usize,
        out_buf: &mut [u8],
    ) -> Result<usize, Error> {
        let mut out_size = 0;
        let res = unsafe {
            wasi_ephemeral_nn::get_output(
                ctx_handle,
                index as i32,
                out_buf.as_mut_ptr() as i32,
                out_buf.len() as i32,
                &mut out_size as *mut _ as i32,
            )
        };

        if res == 0 {
            Ok(out_size)
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }
}

#[cfg(test)]
mod test {
    use super::{ExecutionTarget, GraphBuilder, GraphEncoding};
    use crate::generated;

    #[test]
    fn test_enum_graph_encoding() {
        assert_eq!(GraphEncoding::Openvino as i32, 0);
        assert_eq!(GraphEncoding::Onnx as i32, 1);
        assert_eq!(GraphEncoding::Tensorflow as i32, 2);
        assert_eq!(GraphEncoding::Pytorch as i32, 3);
        assert_eq!(GraphEncoding::TensorflowLite as i32, 4);
    }

    #[test]
    fn test_graph_encoding_with_generated() {
        assert_eq!(
            GraphEncoding::Onnx as i32,
            generated::GRAPH_ENCODING_ONNX.raw() as i32
        );
        assert_eq!(
            GraphEncoding::Openvino as i32,
            generated::GRAPH_ENCODING_OPENVINO.raw() as i32
        );
        assert_eq!(
            GraphEncoding::Pytorch as i32,
            generated::GRAPH_ENCODING_PYTORCH.raw() as i32
        );
        assert_eq!(
            GraphEncoding::Tensorflow as i32,
            generated::GRAPH_ENCODING_TENSORFLOW.raw() as i32
        );
        assert_eq!(
            GraphEncoding::TensorflowLite as i32,
            generated::GRAPH_ENCODING_TENSORFLOWLITE.raw() as i32
        );
    }

    #[test]
    fn test_enum_graph_execution_target() {
        assert_eq!(ExecutionTarget::CPU as i32, 0);
        assert_eq!(ExecutionTarget::GPU as i32, 1);
        assert_eq!(ExecutionTarget::TPU as i32, 2);
    }

    #[test]
    fn test_graph_execution_target_with_generated() {
        assert_eq!(
            ExecutionTarget::CPU as i32,
            generated::EXECUTION_TARGET_CPU.raw() as i32
        );
        assert_eq!(
            ExecutionTarget::GPU as i32,
            generated::EXECUTION_TARGET_GPU.raw() as i32
        );
        assert_eq!(
            ExecutionTarget::TPU as i32,
            generated::EXECUTION_TARGET_TPU.raw() as i32
        );
    }

    #[test]
    fn test_graph_builder() {
        assert_eq!(GraphBuilder::default().encoding, GraphEncoding::Openvino);
        assert_eq!(GraphBuilder::default().target, ExecutionTarget::CPU);

        assert_eq!(GraphBuilder::default().gpu().target, ExecutionTarget::GPU);
        assert_eq!(GraphBuilder::default().tpu().target, ExecutionTarget::TPU);
        assert_eq!(
            GraphBuilder::default().tpu().cpu().target,
            ExecutionTarget::CPU
        );
        assert_eq!(
            GraphBuilder::default()
                .execution_target(ExecutionTarget::GPU)
                .target,
            ExecutionTarget::GPU
        );
    }
}
