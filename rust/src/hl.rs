use crate::generated::*;

pub struct WasiNnGraph {
    execution_ctx: ExecutionContext,
    input_types: TensorTypes,
    output_types: TensorTypes,
}

impl WasiNnGraph {
    pub fn load<'a>(
        data: impl Iterator<Item = &'a [u8]>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Self, NnErrno> {
        unsafe {
            let builders: Vec<&'a [u8]> = data.map(|x| x).collect();

            if builders.len() > 1 {
                let ctx = init_execution_context(load(&builders.as_slice(), encoding, target)?);
                match ctx {
                    Ok(ctx) => Ok(Self {
                        execution_ctx: ExecutionContext::new(ctx),
                        input_types: TensorTypes::new(),
                        output_types: TensorTypes::new(),
                    }),
                    Err(ctx) => Err(ctx),
                }
            } else {
                Err(NN_ERRNO_MISSING_MEMORY)
            }
        }
    }

    pub fn get_execution_context(&self) -> ExecutionContext {
        self.execution_ctx
    }

    pub fn get_input_types(&self) -> &TensorTypes {
        &self.input_types
    }

    pub fn get_output_types(&self) -> &TensorTypes {
        &self.output_types
    }
}

pub struct TensorTypes {
    ttypes: Vec<TensorType>,
}

impl TensorTypes {
    pub fn new() -> Self {
        Self { ttypes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.ttypes.len()
    }
    pub fn get(&self, index: u32) -> Result<TensorType, String> {
        if index < self.ttypes.len() as u32 {
            Ok(self.ttypes[index as usize])
        } else {
            Err(format!("Invalid index {}", index))
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ExecutionContext {
    execution_ctx: GraphExecutionContext,
}

impl ExecutionContext {
    pub fn new(ctx: GraphExecutionContext) -> Self {
        Self { execution_ctx: ctx }
    }

    pub fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), NnErrno> {
        unsafe { set_input(self.execution_ctx, index, tensor) }
    }

    pub fn compute(&mut self) -> Result<(), NnErrno> {
        unsafe { compute(self.execution_ctx) }
    }

    pub fn get_output(
        &self,
        index: u32,
        out_buffer: *mut u8,
        out_buffer_max_size: BufferSize,
    ) -> Result<BufferSize, NnErrno> {
        unsafe { get_output(self.execution_ctx, index, out_buffer, out_buffer_max_size) }
    }
}
