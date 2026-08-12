use crate::tensor::Tensor;
use std::fmt;
use tensor_core::{backend::Backend, dtype::SupportedDType};

impl<B: Backend, T: SupportedDType<B>> fmt::Display for Tensor<B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(backend={}, dtype={}, shape={})",
            B::NAME,
            self.dtype().name(),
            self.shape().dimensions()
        )
    }
}
