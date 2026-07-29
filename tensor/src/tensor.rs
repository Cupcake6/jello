use crate::backend::Backend;
use tensor_core::{backend::TensorOps, dtype::SupportedDType, tensor_metadata::shape::Shape};

type RawTensor<B, T> = <B as TensorOps>::Tensor<T>;

pub struct Tensor<B: Backend, T: SupportedDType<B>>(RawTensor<B, T>);

impl<B: Backend, T: SupportedDType<B>> Tensor<B, T> {
    pub fn full<S: Into<Shape>>(fill_value: T, shape: S) -> Self {
        Self(B::full(fill_value, shape.into()))
    }

    pub fn num_dims(&self) -> usize {
        B::num_dims(&self.0)
    }

    pub fn num_items(&self) -> u64 {
        B::num_items(&self.0)
    }

    pub fn shape(&self) -> &Shape {
        B::shape(&self.0)
    }
}
