use crate::{CpuBackend as B, tensor::CpuTensor};
use std::fmt;
use tensor_core::{
    backend::Backend,
    dtype::SupportedDType,
    tensor::{TensorDisplay, TensorOps},
};

impl<T: SupportedDType<B>> TensorDisplay<T> for CpuTensor<T> {
    fn display(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor(backend={}, dtype={})",
            <B as Backend>::NAME,
            self.dtype().name()
        )
    }
}
