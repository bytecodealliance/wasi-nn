//! Tensor-related definitions.

/// The type of the elements in a tensor.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
#[repr(C)]
pub enum TensorType {
    F16 = 0,
    F32,
    F64,
    U8,
    I32,
    I64,
}

impl TensorType {
    #[inline(always)]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }
}

/// Tensor for wasi-nn syscall, the dimensions is `usize` instead of `u32`.
#[repr(C)]
pub(crate) struct Tensor<'t> {
    pub dimensions: &'t [usize],
    pub tensor_type: TensorType,
    pub data: &'t [u8],
}

impl<'t> Tensor<'t> {
    #[inline(always)]
    pub fn new(dimensions: &'t [usize], tensor_type: TensorType, data: &'t [u8]) -> Self {
        Self {
            dimensions,
            tensor_type,
            data,
        }
    }
}

#[cfg(test)]
mod test {
    use super::TensorType;
    use crate::generated;

    #[test]
    fn test_enum_tensor_type() {
        assert_eq!(TensorType::F16 as u32, 0);
        assert_eq!(TensorType::F32 as u32, 1);
        assert_eq!(TensorType::F64 as u32, 2);
        assert_eq!(TensorType::U8 as u32, 3);
        assert_eq!(TensorType::I32 as u32, 4);
        assert_eq!(TensorType::I64 as u32, 5);
    }

    #[test]
    fn test_tensor_type_with_generated() {
        assert_eq!(
            TensorType::F16 as u32,
            generated::TENSOR_TYPE_F16.raw() as u32
        );
        assert_eq!(
            TensorType::F32 as u32,
            generated::TENSOR_TYPE_F32.raw() as u32
        );
        assert_eq!(
            TensorType::F64 as u32,
            generated::TENSOR_TYPE_F64.raw() as u32
        );
        assert_eq!(
            TensorType::U8 as u32,
            generated::TENSOR_TYPE_U8.raw() as u32
        );
        assert_eq!(
            TensorType::I32 as u32,
            generated::TENSOR_TYPE_I32.raw() as u32
        );
        assert_eq!(
            TensorType::I64 as u32,
            generated::TENSOR_TYPE_I64.raw() as u32
        );
    }

    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    #[test]
    fn test_tensor_tor_wasi_nn_syscall() {
        assert_eq!(std::mem::size_of::<usize>(), std::mem::size_of::<u32>());
        assert_eq!(
            std::mem::size_of::<super::Tensor>(),
            std::mem::size_of::<generated::Tensor>()
        );
    }
}
